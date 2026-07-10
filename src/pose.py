import cv2
import numpy as np

from .stabilize import warp_points
from .config import COURT_LENGTH


def track_and_pose(video_path, pose_model="yolov8s-pose.pt", tracker="bytetrack.yaml",
                   device="cpu", sample_every=1, max_frames=None):
    from ultralytics import YOLO
    model = YOLO(pose_model)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    frames = []
    results = model.track(
        video_path, tracker=tracker, persist=True, device=device,
        stream=True, verbose=False, classes=0,
    )
    fi = 0
    for r in results:
        if fi % sample_every != 0:
            fi += 1
            continue
        if max_frames is not None and len(frames) >= max_frames:
            break
        dets = []
        boxes = r.boxes
        kpts = r.keypoints if r.keypoints is not None else None
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            tid = int(boxes.id[i].item()) if boxes.id is not None else i
            kp = kpts.xy[i].cpu().numpy() if kpts is not None else np.full((17, 2), np.nan)
            dets.append({"id": tid, "bbox": xyxy, "keypoints": kp})
        frames.append(dets)
        fi += 1
    return frames, fps


def foot_point(keypoints):
    kp = np.array(keypoints, dtype=np.float64)
    if kp.ndim == 2 and kp.shape[0] >= 17:
        a = kp[15]
        b = kp[16]
        pts = [p for p in (a, b) if not np.any(np.isnan(p))]
        if pts:
            return np.mean(pts, axis=0)
    if kp.ndim == 2 and kp.shape[0] >= 2:
        return np.nanmean(kp, axis=0)
    return np.array([np.nan, np.nan])


def collect_player_streams(frames, Hs):
    from .stabilize import warp_points
    players = {}
    for fidx, dets in enumerate(frames):
        if fidx >= len(Hs):
            break
        H = Hs[fidx]
        for d in dets:
            pid = d["id"]
            players.setdefault(pid, {"pose_img": [], "pose_court": [], "foot_img": [], "foot_court": []})
            players[pid]["pose_img"].append(d["keypoints"])
            players[pid]["pose_court"].append(warp_points(H, d["keypoints"].reshape(-1, 2)).reshape(17, 2))
            fp = foot_point(d["keypoints"])
            players[pid]["foot_img"].append(fp)
            players[pid]["foot_court"].append(warp_points(H, fp.reshape(1, 2))[0])
    return players


def build_frame_players(frames_all, H, K=None, foot_thresh=1.5, min_track_frac=0.005):
    """Merge fragmented tracker IDs into a small number of stable players.

    If K is given (e.g. 2 for singles, 4 for doubles), fragment tracks are
    clustered by floor position and merged into exactly K players, so one
    physical player is not reported as several. Otherwise up to 6 tracks are
    kept (legacy behaviour).

    Returns a dict pid -> {"pose_court":(N,17,2), "foot_court":(N,2),
    "pose_img":(N,17,2), "racket":(N,2)} all indexed by VIDEO FRAME (NaN where
    the player is not detected). All coordinates are warped with the single
    stable global homography H (meters), which removes per-frame jitter.
    """
    from .racket_bootstrap import racket_tip_court

    N = len(frames_all)
    dets = []
    for f, dlist in enumerate(frames_all):
        for d in dlist:
            kp = np.array(d["keypoints"], dtype=np.float64)
            if kp.ndim != 2 or np.all(np.isnan(kp)):
                continue
            pc = warp_points(H, kp.reshape(-1, 2)).reshape(17, 2)
            ft = foot_point(kp)
            ftc = warp_points(H, ft.reshape(1, 2))[0]
            if np.any(np.isnan(ftc)):
                continue
            dets.append((f, ftc, kp, pc))

    tracks = []
    for f, ftc, kp, pc in dets:
        best_i, best_d = -1, np.inf
        for i, t in enumerate(tracks):
            dd = float(np.linalg.norm(t["last"] - ftc))
            if dd < best_d:
                best_d, best_i = dd, i
        if best_i >= 0 and best_d <= foot_thresh:
            t = tracks[best_i]
        else:
            t = {"last": ftc, "frames": [], "pose_img": [], "pose_court": [], "foot": []}
            tracks.append(t)
        t["frames"].append(f)
        t["pose_img"].append(kp)
        t["pose_court"].append(pc)
        t["foot"].append(ftc)
        t["last"] = ftc

    tracks.sort(key=lambda t: len(t["frames"]), reverse=True)
    if K is None:
        maxc = max((len(t["frames"]) for t in tracks), default=0)
        keep = [t for t in tracks if len(t["frames"]) >= max(10, 0.3 * maxc)]
        keep = keep[:6]
        if len(keep) < 2:
            keep = tracks[:2]
    else:
        # Known player count (singles/doubles): drop tiny fragments, then merge
        # the rest into exactly K players by floor position.
        cand = [t for t in tracks if len(t["frames"]) >= 10] or tracks
        keep = _cluster_tracks_to_k(cand, K)

    players = {}
    for pi, t in enumerate(keep):
        pose_court = np.full((N, 17, 2), np.nan)
        foot_court = np.full((N, 2), np.nan)
        pose_img = np.full((N, 17, 2), np.nan)
        racket = np.full((N, 2), np.nan)
        for f, kp, pc, ftc in zip(t["frames"], t["pose_img"], t["pose_court"], t["foot"]):
            pose_court[f] = pc
            foot_court[f] = ftc
            pose_img[f] = kp
            racket[f] = racket_tip_court(kp, H)
        players[pi] = {
            "pose_court": pose_court,
            "foot_court": foot_court,
            "pose_img": pose_img,
            "racket": racket,
        }
    return players


def _merge_frag_tracks(frag_tracks):
    """Combine several fragment tracks (same physical player) into one, keeping
    per-frame order."""
    merged = {"frames": [], "pose_img": [], "pose_court": [], "foot": [], "last": None}
    for t in frag_tracks:
        merged["frames"].extend(t["frames"])
        merged["pose_img"].extend(t["pose_img"])
        merged["pose_court"].extend(t["pose_court"])
        merged["foot"].extend(t["foot"])
    order = np.argsort(np.array(merged["frames"]))
    merged["frames"] = [merged["frames"][i] for i in order]
    merged["pose_img"] = [merged["pose_img"][i] for i in order]
    merged["pose_court"] = [merged["pose_court"][i] for i in order]
    merged["foot"] = [merged["foot"][i] for i in order]
    merged["last"] = merged["foot"][-1] if merged["foot"] else np.array([np.nan, np.nan])
    return merged


def _cluster_tracks_to_k(tracks, K, seed=0):
    """Merge fragmented tracker IDs into exactly K players.

    For singles (K=2) players defend opposite halves, separated by the net line
    (y = COURT_LENGTH/2). Assign each fragment to a half by its mean foot-y and
    merge within the half -- deterministic and robust to fragmentation (K-means
    on centroids can collapse into a bad local minimum, mixing both halves into
    each player). For K != 2 fall back to K-means on floor centroids.
    """
    if len(tracks) <= K:
        return tracks
    if K == 2:
        mid = COURT_LENGTH / 2.0
        groups = [[], []]
        for t in tracks:
            fy = np.nanmean(np.array(t["foot"], dtype=np.float64), axis=0)[1]
            groups[0 if fy < mid else 1].append(t)
        if not groups[0]:
            groups[0], groups[1] = groups[1], groups[0]
        return [_merge_frag_tracks(g) for g in groups if g]
    cents = np.array(
        [np.nanmean(np.array(t["foot"], dtype=np.float64), axis=0) for t in tracks])
    rng = np.random.default_rng(seed)
    idx0 = rng.integers(len(cents))
    centers = [cents[idx0]]
    for _ in range(1, K):
        d = np.min([np.sum((cents - c) ** 2, axis=1) for c in centers], axis=0)
        if d.sum() == 0:
            d = np.ones(len(cents))
        probs = d / d.sum()
        centers.append(cents[rng.choice(len(cents), p=probs)])
    centers = np.array(centers, dtype=np.float64)
    for _ in range(20):
        dist = np.stack([np.sum((cents - c) ** 2, axis=1) for c in centers])
        labels = np.argmin(dist, axis=0)
        for k in range(K):
            m = labels == k
            if m.any():
                centers[k] = cents[m].mean(axis=0)
    out = []
    for k in range(K):
        grp = [tracks[i] for i in range(len(tracks)) if labels[i] == k]
        out.append(_merge_frag_tracks(grp))
    return out


def load_pose_model(pose_model="yolov8s-pose.pt", device="cpu"):
    from ultralytics import YOLO
    return YOLO(pose_model)


def parse_detections(result):
    dets = []
    if result is None:
        return dets
    boxes = result.boxes
    kpts = result.keypoints if result.keypoints is not None else None
    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].cpu().numpy()
        tid = int(boxes.id[i].item()) if boxes.id is not None else i
        kp = kpts.xy[i].cpu().numpy() if kpts is not None else np.full((17, 2), np.nan)
        dets.append({"id": tid, "bbox": xyxy, "keypoints": kp})
    return dets


def track_frame(model, frame, device="cpu", tracker="bytetrack.yaml"):
    results = model.track(frame, tracker=tracker, persist=True, device=device,
                          verbose=False, classes=0)
    if isinstance(results, (list, tuple)):
        results = results[0]
    return parse_detections(results)
