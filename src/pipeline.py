import csv
import json
import os

import numpy as np
import torch

from . import stabilize, pose as posemod, shuttle as shuttlemod, contact as contactmod
from . import biomech, racket_bootstrap, movement, baseline, viz, llm_feedback, ab_eval
from . import inpaintnet as inpaintmod
from .config import STROKE_TO_ID, canonical_stroke, COURT_LENGTH, COURT_WIDTH, validate_court_corners


def _wrist_stream(players, pid):
    seq = []
    for p in players[pid]["pose_court"]:
        w = p[9] if not np.any(np.isnan(p[9])) else p[10]
        seq.append(w)
    return np.array(seq, dtype=np.float64)


def run_full_pipeline(video, corners, out_dir="data", labels_csv=None,
                      device="cpu", tracknet_weights=None, inpaintnet_weights=None,
                      use_mbh=False, llm_provider=None, llm_key=None, max_frames=None,
                      batch_size=128, debug=False):
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    if str(device) == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to cpu")
        device = "cpu"
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print("[1-2/8] batched stabilize + track + pose (batch_size=%d)..." % batch_size)
    validate_court_corners(corners)
    state = stabilize.init_stabilizer_state(corners)
    H0 = state["H0"]
    model = posemod.load_pose_model("yolov8s-pose.pt", device=device)
    tracknet = shuttlemod.TrackNetShuttle(tracknet_weights, device=device) if tracknet_weights else None
    if tracknet is None:
        print("WARNING: TrackNet weights NOT provided (tracknet_weights=None). "
              "Shuttle detection is disabled -> contacts=0 and stroke classification "
              "cannot run. Pass the trained TrackNet .pt weights to enable it.")
    inpaintnet = inpaintmod.load_inpaintnet(inpaintnet_weights, device=device)

    Hs = []
    frames_all = []
    shuttle_img = []
    frame_hw = None
    n_read = 0
    cap_max = max_frames if max_frames is not None else float("inf")
    while n_read < cap_max:
        batch = []
        while len(batch) < batch_size and n_read < cap_max:
            ret, f = cap.read()
            if not ret:
                break
            batch.append(f)
            n_read += 1
        if not batch:
            break
        for f in batch:
            if frame_hw is None:
                frame_hw = f.shape[:2]
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            Hs.append(stabilize.stabilize_frame(state, gray))
        for f in batch:
            frames_all.append(posemod.track_frame(model, f, device=device))
        if tracknet is not None:
            shuttle_img.extend(tracknet.predict_frames(batch))
        else:
            shuttle_img.extend([[np.nan, np.nan]] * len(batch))
        del batch
    cap.release()
    Hs = np.array(Hs)
    print("  processed %d frames" % len(Hs))

    print("  consolidating tracker fragments into players (stable global H)...")
    players = posemod.build_frame_players(frames_all, H0)

    print("[3/8] shuttle tracking + contact detection...")
    shuttle_px = np.array(shuttle_img, dtype=np.float64)
    if inpaintnet is not None and frame_hw is not None:
        shuttle_px = inpaintmod.rectify_trajectory(
            shuttle_px, frame_hw[1], frame_hw[0], inpaintnet, device=device, seq_len=16)
        print("  InpaintNet: trajectory rectified (%d frames)" % len(shuttle_px))
    shuttle_court = np.full((len(Hs), 2), np.nan)
    for i in range(len(Hs)):
        if i < len(shuttle_px) and not np.any(np.isnan(np.array(shuttle_px[i], dtype=np.float64))):
            shuttle_court[i] = stabilize.warp_points(H0, np.array(shuttle_px[i], dtype=np.float64).reshape(1, 2))[0]
    # Mask shuttle detections outside the court (false positives / teleports).
    oob = ((shuttle_court[:, 0] < -0.5) | (shuttle_court[:, 0] > COURT_WIDTH + 0.5) |
           (shuttle_court[:, 1] < -0.5) | (shuttle_court[:, 1] > COURT_LENGTH + 0.5))
    shuttle_court[oob] = np.nan
    contacts = contactmod.detect_contacts_near_players(
        shuttle_court, {p: players[p]["pose_court"] for p in players}, fps, max_dist=2.0)
    if len(contacts) == 0:
        contacts = contactmod.detect_contact_frames(shuttle_court, fps, angle_thresh_deg=50.0, min_speed=1.0)
    cov = int(np.sum(~np.isnan(shuttle_court).any(axis=1)))
    spd = np.linalg.norm(contactmod.shuttle_speed(shuttle_court, fps), axis=1)
    shi = np.array(shuttle_img, dtype=np.float64)
    print(f"[shuttle] frames={len(Hs)} shuttle_nonnan={cov} "
          f"({100*cov/max(len(Hs),1):.1f}%) contacts={len(contacts)} "
          f"speed_m/s min={np.nanmin(spd):.2f} med={np.nanmedian(spd):.2f} "
          f"max={np.nanmax(spd):.2f} frac>1m/s={100*float(np.nanmean(spd > 1.0)):.1f}%")
    print(f"[shuttle] raw_img x[{np.nanmin(shi[:, 0]):.0f},{np.nanmax(shi[:, 0]):.0f}] "
          f"y[{np.nanmin(shi[:, 1]):.0f},{np.nanmax(shi[:, 1]):.0f}] "
          f"court x[{np.nanmin(shuttle_court[:, 0]):.2f},{np.nanmax(shuttle_court[:, 0]):.2f}] "
          f"y[{np.nanmin(shuttle_court[:, 1]):.2f},{np.nanmax(shuttle_court[:, 1]):.2f}]")
    if debug and contacts:
        print("[debug] first contacts:", contacts[:20])
    if len(contacts) == 0:
        print("WARNING: 0 shuttle contacts detected. Check TrackNet weights and shuttle coverage.")
    attrib = biomech.attribute_contact(contacts, {p: players[p]["pose_court"] for p in players}, shuttle_court)

    print("[4/8] racket trajectories...")
    racket_streams = {p: players[p]["racket"] for p in players}

    print("[5/8] classification...")
    foot_streams = {p: players[p]["foot_court"] for p in players}
    wrist_streams = {p: _wrist_stream(players, p) for p in players}
    preds = baseline.build_baseline_predictions(contacts, attrib, foot_streams, wrist_streams, shuttle_court, fps)

    frame_to_shot = None
    if labels_csv and os.path.exists(labels_csv):
        frame_to_shot = _label_frame_map(labels_csv)
        print("  training fusion classifier on labeled shots...")
        try:
            trained = _train_and_predict(labels_csv, contacts, attrib, players, racket_streams, Hs, fps, device, debug=debug)
            if trained is None:
                print("  too few matching labeled shots; keeping geometry baseline predictions")
            else:
                preds = trained
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("  classifier training failed, using baseline:", e)

    _write_predictions(out_dir, contacts, attrib, preds, frame_to_shot)

    print("[6/8] movement + fatigue analytics...")
    mv = movement.compute_movement(foot_streams, fps, max_step=0.8)
    hm = movement.court_heatmap(foot_streams)
    fat = movement.fatigue_profile(foot_streams, fps, max_step=0.8)
    metrics = {
        "movement": {p: {"total_distance_m": mv[p]["total_distance_m"],
                         "mean_speed": mv[p]["mean_speed"],
                         "max_speed": mv[p]["max_speed"]} for p in mv},
        "fatigue": fat,
        "n_contacts": len(contacts),
        "stroke_counts": _stroke_counts(preds),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("[7/8] visualization...")
    viz.plot_heatmap(hm, os.path.join(out_dir, "coverage_heatmap.png"))
    viz.plot_fatigue(fat, os.path.join(out_dir, "fatigue.png"))
    viz.draw_annotated_video(video, frames_all, Hs, shuttle_img, contacts, attrib, preds,
                             os.path.join(out_dir, "annotated.mp4"))

    print("[8/8] report...")
    report_path = None
    if llm_provider and llm_key:
        text = llm_feedback.generate_feedback(metrics, llm_provider, llm_key)
        report_path = llm_feedback.write_report(metrics, text, os.path.join(out_dir, "coaching_report.md"))
    else:
        report_path = llm_feedback.write_report(metrics, "(LLM feedback disabled; set llm_provider/key)",
                                                os.path.join(out_dir, "coaching_report.md"))

    print("Done. Outputs in", out_dir)
    return {"metrics": metrics, "predictions_csv": os.path.join(out_dir, "new_predictions.csv"),
            "report": report_path}


def _train_and_predict(labels_csv, contacts, attrib, players, racket_streams, Hs, fps, device, debug=False):
    from . import classifier as clfmod
    import csv as _csv
    gt = [r for r in _csv.DictReader(open(labels_csv)) if r.get("label_status") == "labeled"]
    frame_to_label = {}
    for g in gt:
        try:
            frame_to_label[int(float(g["frame"]))] = canonical_stroke(g["true_stroke"])
        except Exception:
            pass
    if debug:
        in_range = [f for f in frame_to_label if f < len(Hs)]
        print(f"[debug] labeled_shots={len(frame_to_label)} in_processed_range={len(in_range)}")
        print(f"[debug] contacts(detected)={len(contacts)}")
    court_poses = {p: players[p]["pose_court"] for p in players}
    mbh_dummy = np.zeros((len(Hs), 1))
    MATCH_WINDOW = 15

    # Real detected contacts -> windows we will evaluate/predict on.
    real_samples = clfmod.extract_stroke_windows(court_poses, racket_streams, mbh_dummy, contacts, attrib)

    # Label the real samples that match a labeled shot.
    train = []
    used_labels = set()
    for s in real_samples:
        near = min(frame_to_label.keys(), key=lambda k: abs(k - s["contact"])) if frame_to_label else None
        if near is not None and abs(near - s["contact"]) <= MATCH_WINDOW:
            s["label"] = STROKE_TO_ID[frame_to_label[near]]
            train.append(s)
            used_labels.add(near)

    # Boost training data: synthesize a contact at every labeled shot frame
    # that was not already matched to a detected contact.
    if frame_to_label:
        for lf in sorted(frame_to_label):
            if lf in used_labels or lf >= len(Hs):
                continue
            if any(abs(lf - c) <= MATCH_WINDOW for c in contacts):
                continue
            pa = biomech.attribute_contact([lf], court_poses, shuttle_court=None, player_ids=list(players.keys()))
            ps = clfmod.extract_stroke_windows(court_poses, racket_streams, mbh_dummy, [lf], pa)
            for s in ps:
                s["label"] = STROKE_TO_ID[frame_to_label[lf]]
                train.append(s)

    if debug:
        def _matches(w):
            return sum(1 for f in frame_to_label
                       if f < len(Hs) and any(abs(f - c) <= w for c in contacts))
        print(f"[debug] real_samples={len(real_samples)} train_samples={len(train)} "
              f"labeled_matched@3={_matches(3)} matched@10={_matches(10)} matched@15={_matches(15)}")

    if len(train) < 5:
        return None
    model = clfmod.train_classifier(train, [], len(STROKE_TO_ID), device=device)
    idx = clfmod.predict_classifier(model, real_samples, device)
    from .config import CANONICAL_STROKES
    return [CANONICAL_STROKES[i] for i in idx]


def _write_predictions(out_dir, contacts, attrib, preds, frame_to_shot=None):
    path = os.path.join(out_dir, "new_predictions.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shot_id", "frame", "player_id", "predicted_stroke", "predicted_class_id", "source"])
        for i, (cf, pid, pr) in enumerate(zip(contacts, attrib, preds)):
            if frame_to_shot:
                sid = min(frame_to_shot, key=lambda k: abs(k - cf))
                sid = frame_to_shot[sid] if abs(sid - cf) <= 3 else (i + 1)
            else:
                sid = i + 1
            w.writerow([sid, cf, pid, pr, STROKE_TO_ID.get(canonical_stroke(pr), -1), "pipeline"])


def _label_frame_map(labels_csv):
    import csv as _csv
    m = {}
    for r in _csv.DictReader(open(labels_csv)):
        if r.get("label_status") == "labeled":
            try:
                m[int(float(r["frame"]))] = int(float(r["shot_id"]))
            except Exception:
                pass
    return m


def _stroke_counts(preds):
    from collections import Counter
    c = Counter(preds)
    return dict(c)


def ab_compare(labels_csv, new_preds_csv, bst_preds_csv=None):
    return ab_eval.run_ab(labels_csv, new_preds_csv, bst_preds_csv)
