import numpy as np

from .config import PLAYER_IDS, COURT_LENGTH

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


def attribute_contact(contact_frames, poses_court, shuttle_court, player_ids=None,
                      window=3, max_dist=2.0, debug=False, half_aware=True,
                      half_tol=1.0, half_gate=4.0):
    """Assign each contact frame to the player who hit.

    Attribution priority:
    - **Nearest wrist within `max_dist`** is the primary cue. At a real contact
      the shuttle sits next to the hitting player's wrist, so this is the most
      robust signal and must win even if the (noisy, extrapolated) court half
      disagrees -- a bad shuttle detection warps to the wrong half and would
      otherwise mis-attribute (see side_agreement regression when half led).
    - **Half-aware fallback** (`half_aware=True`, default): only when no player
      is within `max_dist`, if the shuttle is clearly on one half and that half's
      player has a wrist within `half_gate` m, attribute to that player. Tolerant
      of warp noise when the shuttle is genuinely between/away from both players.
      Near the net it falls back to the nearest wrist.
    - `half_aware=False`: plain nearest-wrist within `max_dist` (legacy).

    A contact is dropped (None) when no valid wrist is within range."""
    if player_ids is None:
        player_ids = list(poses_court.keys())

    # Stable per-player half from mean foot (ankle) y. Far = smaller y (top of
    # image), near = larger y (toward camera).
    half_of = {}
    for pid in player_ids:
        pseq = poses_court.get(pid)
        if pseq is None:
            continue
        foot = pseq[:, 15:17, 1]  # ankle y over frames
        foot = foot[~np.any(np.isnan(foot), axis=1)]
        if len(foot):
            half_of[pid] = "near" if foot.mean() > COURT_LENGTH / 2 else "far"

    net = COURT_LENGTH / 2
    attrib = []
    n_dropped = 0
    for cf in contact_frames:
        if shuttle_court is not None and (cf >= len(shuttle_court) or np.any(np.isnan(shuttle_court[cf]))):
            attrib.append(None)
            if debug:
                print(f"[attrib] cf={cf} -> NONE (shuttle NaN/OOB)")
            continue
        lo = max(0, cf - window)
        if shuttle_court is not None:
            hi = min(len(shuttle_court), cf + window + 1)
        else:
            hi = cf + window + 1
        target = None if shuttle_court is None else shuttle_court[cf]
        best_pid, best_d = None, np.inf
        per_pid = {}
        for pid in player_ids:
            pseq = poses_court.get(pid)
            if pseq is None or len(pseq) <= lo:
                continue
            sub = pseq[lo:hi]
            pmin = np.inf
            for pose in sub:
                for widx in WRIST:
                    w = pose[widx]
                    if np.any(np.isnan(w)):
                        continue
                    d = 0.0 if target is None else float(np.linalg.norm(w - target))
                    pmin = min(pmin, d)
            if pmin < np.inf:
                per_pid[pid] = pmin
                if pmin < best_d:
                    best_d, best_pid = pmin, pid

        chosen = None
        mode = "NEAR"
        # Primary signal: nearest wrist within max_dist. At a real contact the
        # shuttle sits next to the hitting player's wrist, so this is the most
        # robust cue and must NOT be overridden by the (noisy, extrapolated) court
        # half -- a bad shuttle detection warps to the wrong half and half-aware
        # would then mis-attribute. See side_agreement regression when half led.
        if best_pid is not None and best_d <= max_dist:
            chosen = best_pid
            mode = "NEAR"
        # Fallback: no player within max_dist, but the shuttle is clearly on one
        # half and that half's player has a wrist within half_gate -> use the half
        # (tolerant of warp noise when the shuttle is genuinely away from both
        # players but on one side). Near the net, use the nearest wrist.
        elif half_aware and target is not None and half_of:
            sy = target[1]
            if sy < net - half_tol:
                cand = [p for p, h in half_of.items() if h == "far"]
            elif sy > net + half_tol:
                cand = [p for p, h in half_of.items() if h == "near"]
            else:
                cand = list(half_of.keys())  # near net: use nearest wrist
            cand_valid = [(p, per_pid[p]) for p in cand if p in per_pid]
            if cand_valid:
                cand_valid.sort(key=lambda kv: kv[1])
                cpid, cd = cand_valid[0]
                if cd <= half_gate:
                    chosen = cpid
                    mode = "HALF" if (sy < net - half_tol or sy > net + half_tol) else "NEAR"
        if chosen is not None:
            attrib.append(chosen)
        else:
            attrib.append(None)
            n_dropped += 1
        if debug:
            dists = " ".join(f"p{pid}:{d:.2f}" for pid, d in sorted(per_pid.items()))
            verdict = f"pid={chosen} dist={per_pid.get(chosen, best_d):.2f}" if chosen is not None else "NO-WRIST"
            tag = "" if chosen is not None else " DROPPED"
            print(f"[attrib] cf={cf} -> {verdict} [{mode}] | {dists}{tag}")
    if debug:
        print(f"[attrib] summary: dropped={n_dropped}/{len(contact_frames)} "
              f"(max_dist={max_dist}m half_gate={half_gate}m)")
    return attrib
