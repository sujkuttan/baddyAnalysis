import numpy as np

from .config import PLAYER_IDS

SHOULDER = [5, 6]
ELBOW = [7, 8]
WRIST = [9, 10]


def _limb_velocity(court_poses, idx, fps):
    seq = np.array([p[idx] for p in court_poses], dtype=np.float64)
    v = np.full((len(seq), 2), np.nan)
    if len(seq) < 2:
        return v
    v[1:] = (seq[1:] - seq[:-1]) * fps
    v[0] = v[1]
    return v


def kinetic_chain(court_poses, contact_frame, fps, window=14):
    lo = max(0, contact_frame - window)
    hi = min(len(court_poses), contact_frame + window + 1)
    sub = court_poses[lo:hi]
    if len(sub) < 5:
        return {"order": None, "fluidity": 0.0, "wrist_speed": 0.0}
    vs = _limb_velocity(sub, SHOULDER[0], fps)
    ve = _limb_velocity(sub, ELBOW[0], fps)
    vw = _limb_velocity(sub, WRIST[0], fps)
    sp = np.linalg.norm(vs, axis=1)
    ep = np.linalg.norm(ve, axis=1)
    wp = np.linalg.norm(vw, axis=1)

    def peak(a):
        if np.all(np.isnan(a)):
            return -1
        return int(np.nanargmax(a))

    ps, pe, pw = peak(sp), peak(ep), peak(wp)
    order_ok = (ps <= pe <= pw) and ps >= 0
    wrist_speed = float(np.nanmax(wp)) if len(wp) else 0.0
    return {
        "order": "->".join(["shoulder", "elbow", "wrist"] if order_ok else ["?"]),
        "sequential": bool(order_ok),
        "wrist_speed": wrist_speed,
        "shoulder_peak": ps,
        "elbow_peak": pe,
        "wrist_peak": pw,
    }


def attribute_contact(contact_frames, poses_court, shuttle_court, player_ids=None):
    if player_ids is None:
        player_ids = PLAYER_IDS
    attrib = []
    for cf in contact_frames:
        if cf >= len(shuttle_court) or np.any(np.isnan(shuttle_court[cf])):
            attrib.append(None)
            continue
        best, best_d = None, np.inf
        for pid in player_ids:
            if pid not in poses_court:
                continue
            if cf >= len(poses_court[pid]):
                continue
            pose = poses_court[pid][cf]
            for widx in WRIST:
                w = pose[widx]
                if np.any(np.isnan(w)):
                    continue
                d = np.linalg.norm(w - shuttle_court[cf])
                if d < best_d:
                    best_d, best = d, pid
        attrib.append(best)
    return attrib
