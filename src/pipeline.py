import csv
import json
import os

import numpy as np

from . import stabilize, pose as posemod, shuttle as shuttlemod, contact as contactmod
from . import biomech, racket_bootstrap, movement, baseline, viz, llm_feedback, ab_eval
from .config import STROKE_TO_ID, canonical_stroke, COURT_LENGTH


def _wrist_stream(players, pid):
    seq = []
    for p in players[pid]["pose_court"]:
        w = p[9] if not np.any(np.isnan(p[9])) else p[10]
        seq.append(w)
    return np.array(seq, dtype=np.float64)


def run_full_pipeline(video, corners, out_dir="data", labels_csv=None,
                      device="cpu", tracknet_weights=None, use_mbh=False,
                      llm_provider=None, llm_key=None):
    os.makedirs(out_dir, exist_ok=True)
    print("[1/8] stabilizing camera (per-frame homography)...")
    st = stabilize.stabilize_video(video, corners)
    Hs, fps = st["homographies"], st["fps"]

    print("[2/8] tracking + pose...")
    frames, _ = posemod.track_and_pose(video, device=device)
    players = posemod.collect_player_streams(frames, Hs)

    print("[3/8] shuttle tracking + contact detection...")
    shuttle_court = np.full((len(Hs), 2), np.nan)
    try:
        sh = shuttlemod.TrackNetShuttle(tracknet_weights, device=device)
        shuttle_img = sh.predict_video(video)
        for i in range(min(len(shuttle_img), len(Hs))):
            if not np.any(np.isnan(shuttle_img[i])):
                shuttle_court[i] = stabilize.warp_points(Hs[i], shuttle_img[i].reshape(1, 2))[0]
    except Exception as e:
        print("  shuttle tracking skipped:", e)
    contacts = contactmod.detect_contact_frames(shuttle_court, fps)
    attrib = biomech.attribute_contact(contacts, {p: players[p]["pose_court"] for p in players}, shuttle_court)

    print("[4/8] racket trajectories...")
    racket_streams = {}
    for pid in players:
        racket_streams[pid] = racket_bootstrap.racket_trajectory(players[pid]["pose_img"], Hs)

    print("[5/8] classification...")
    foot_streams = {p: np.array(players[p]["foot_court"]) for p in players}
    wrist_streams = {p: _wrist_stream(players, p) for p in players}
    preds = baseline.build_baseline_predictions(contacts, attrib, foot_streams, wrist_streams, shuttle_court, fps)

    frame_to_shot = None
    if labels_csv and os.path.exists(labels_csv):
        frame_to_shot = _label_frame_map(labels_csv)
        print("  training fusion classifier on labeled shots...")
        try:
            preds = _train_and_predict(labels_csv, contacts, attrib, players, racket_streams, Hs, fps, device)
        except Exception as e:
            print("  classifier training failed, using baseline:", e)

    _write_predictions(out_dir, contacts, attrib, preds, frame_to_shot)

    print("[6/8] movement + fatigue analytics...")
    mv = movement.compute_movement(foot_streams, fps)
    hm = movement.court_heatmap(foot_streams)
    fat = movement.fatigue_profile(foot_streams, fps)
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
    viz.draw_annotated_video(video, frames, Hs, shuttle_court, contacts, attrib, preds,
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


def _train_and_predict(labels_csv, contacts, attrib, players, racket_streams, Hs, fps, device):
    from . import classifier as clfmod
    import csv as _csv
    gt = [r for r in _csv.DictReader(open(labels_csv)) if r.get("label_status") == "labeled"]
    frame_to_label = {}
    for g in gt:
        try:
            frame_to_label[int(float(g["frame"]))] = canonical_stroke(g["true_stroke"])
        except Exception:
            pass
    court_poses = {p: np.array(players[p]["pose_court"]) for p in players}
    mbh_dummy = np.zeros((len(Hs), 1))
    samples_all = clfmod.extract_stroke_windows(court_poses, racket_streams, mbh_dummy, contacts, attrib)
    train, val = [], []
    for s in samples_all:
        near = min(frame_to_label.keys(), key=lambda k: abs(k - s["contact"])) if frame_to_label else None
        if near is not None and abs(near - s["contact"]) <= 3:
            s["label"] = STROKE_TO_ID[frame_to_label[near]]
            (train if len(train) < len(frame_to_label) * 0.8 else val).append(s)
    if len(train) < 5:
        return None
    model = clfmod.train_classifier(train, val, len(STROKE_TO_ID), device=device)
    idx = clfmod.predict_classifier(model, samples_all, device)
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
