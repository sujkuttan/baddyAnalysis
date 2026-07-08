import cv2
import numpy as np

from .stabilize import warp_points


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
        keep = [t for t in tracks if len(t["frames"]) >= max(10, int(min_track_frac * N))]
        if len(keep) < 2:
            keep = tracks[:2]
    else:
        keep = tracks[:K]

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
