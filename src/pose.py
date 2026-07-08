import cv2
import numpy as np


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
