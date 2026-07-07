import numpy as np

from .stabilize import warp_points

COCO = {
    "elbow_l": 7, "wrist_l": 9,
    "elbow_r": 8, "wrist_r": 10,
}


def _arm_tip(elbow, wrist, extend=1.5):
    elbow = np.array(elbow, dtype=np.float64)
    wrist = np.array(wrist, dtype=np.float64)
    d = wrist - elbow
    n = np.linalg.norm(d)
    if n < 1e-6:
        return wrist
    return wrist + d / n * (n * (extend - 1.0))


def racket_tips_image(pose_img):
    pose_img = np.array(pose_img, dtype=np.float64)
    tips = {}
    for side in ("l", "r"):
        e = pose_img[COCO["elbow_" + side]]
        w = pose_img[COCO["wrist_" + side]]
        if np.any(np.isnan(e)) or np.any(np.isnan(w)):
            tips[side] = np.array([np.nan, np.nan])
        else:
            tips[side] = _arm_tip(e, w)
    return tips


def racket_tip_court(pose_img, H, extend=1.5):
    tips_img = racket_tips_image(pose_img)
    out = {}
    for side, t in tips_img.items():
        if np.any(np.isnan(t)):
            out[side] = np.array([np.nan, np.nan])
        else:
            out[side] = warp_points(H, t.reshape(1, 2))[0]
    if not np.any(np.isnan(out.get("r", [np.nan, np.nan]))):
        return out["r"]
    if not np.any(np.isnan(out.get("l", [np.nan, np.nan]))):
        return out["l"]
    return np.array([np.nan, np.nan])


def racket_trajectory(poses_img, Hs):
    traj = []
    for pose, H in zip(poses_img, Hs):
        if pose is None or H is None:
            traj.append(np.array([np.nan, np.nan]))
        else:
            traj.append(racket_tip_court(pose, H))
    return np.array(traj, dtype=np.float64)
