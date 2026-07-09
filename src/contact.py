import numpy as np


def shuttle_speed(shuttle, fps):
    shuttle = np.array(shuttle, dtype=np.float64)
    v = np.full((len(shuttle), 2), np.nan)
    if len(shuttle) < 2:
        return v
    d = shuttle[1:] - shuttle[:-1]
    v[1:] = d * fps
    v[0] = v[1]
    return v


def detect_contacts_image_space(shuttle_px, pose_img_streams, fps,
                                 max_dist_px=60.0, min_gap=0.15, win=3):
    """Contact cue in raw image space: a hit is the shuttle nearest a player's
    wrist (keypoints 9/10) in pixels. Independent of the court homography, so it
    recovers hits where the warped shuttle landed far off-court (aerial lobs).
    """
    shuttle = np.array(shuttle_px, dtype=np.float64)
    N = len(shuttle)
    dist = np.full(N, np.inf)
    for pseq in pose_img_streams.values():
        pseq = np.array(pseq, dtype=np.float64)
        if pseq.ndim != 3 or pseq.shape[0] != N:
            continue
        for widx in (9, 10):
            w = pseq[:, widx]
            valid = ~np.any(np.isnan(w), axis=1)
            d = np.full(N, np.inf)
            d[valid] = np.linalg.norm(w[valid] - shuttle[valid], axis=1)
            dist = np.minimum(dist, d)
    contacts = []
    for i in range(win, N - win):
        if np.isnan(dist[i]) or np.isnan(shuttle[i]).any():
            continue
        if dist[i] > max_dist_px:
            continue
        lo = max(0, i - win)
        hi = min(N, i + win + 1)
        if dist[i] <= dist[lo:hi].min():
            contacts.append(i)
    dedup = []
    last = -100
    gap = int(min_gap * fps)
    for c in contacts:
        if c - last > gap:
            dedup.append(c)
            last = c
    return dedup


def merge_contacts(*lists, min_gap=0.15, fps=30):
    """Union several contact-frame lists and dedupe by a minimum frame gap."""
    merged = sorted(set().union(*[set(l) for l in lists if l]))
    dedup = []
    last = -100
    gap = int(min_gap * fps)
    for c in merged:
        if c - last > gap:
            dedup.append(c)
            last = c
    return dedup


def detect_contact_frames(shuttle, fps, angle_thresh_deg=70.0, min_speed=2.0):
    shuttle = np.array(shuttle, dtype=np.float64)
    speed = shuttle_speed(shuttle, fps)
    spd = np.linalg.norm(speed, axis=1)
    contacts = []
    for i in range(2, len(shuttle) - 2):
        if np.isnan(spd[i]):
            continue
        pre = speed[i - 2]
        post = speed[i + 1]
        if np.isnan(pre).any() or np.isnan(post).any():
            continue
        npre, npost = np.linalg.norm(pre), np.linalg.norm(post)
        if min(npre, npost) < min_speed:
            continue
        cosang = np.dot(pre, post) / (npre * npost + 1e-6)
        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        if ang > angle_thresh_deg:
            contacts.append(i)
    dedup = []
    last = -100
    for c in contacts:
        if c - last > int(0.15 * fps):
            dedup.append(c)
            last = c
    return dedup


def segment_rallies(contact_frames, fps, gap_sec=4.0):
    if len(contact_frames) == 0:
        return []
    rallies = []
    start = contact_frames[0]
    prev = contact_frames[0]
    for c in contact_frames[1:]:
        if (c - prev) / fps > gap_sec:
            rallies.append((start, prev))
            start = c
        prev = c
    rallies.append((start, prev))
    return rallies


def detect_contacts_near_players(shuttle, poses_court, fps, max_dist=2.0, min_gap=0.15, win=3):
    """Detect hits as local minima of shuttle->nearest-player-wrist distance.

    A shuttle in flight follows a smooth parabola (no abrupt direction
    reversal), so angle-based detection misses hits. The physically correct
    cue is the shuttle being closest to a player's hand at the moment of contact.
    """
    shuttle = np.array(shuttle, dtype=np.float64)
    N = len(shuttle)
    dist = np.full(N, np.inf)
    for pid, pseq in poses_court.items():
        pseq = np.array(pseq, dtype=np.float64)
        if pseq.ndim != 3 or pseq.shape[0] != N:
            continue
        for widx in (9, 10):
            w = pseq[:, widx]
            valid = ~np.any(np.isnan(w), axis=1)
            d = np.full(N, np.inf)
            d[valid] = np.linalg.norm(w[valid] - shuttle[valid], axis=1)
            dist = np.minimum(dist, d)
    contacts = []
    for i in range(win, N - win):
        if np.isnan(dist[i]) or np.isnan(shuttle[i]).any():
            continue
        if dist[i] > max_dist:
            continue
        lo = max(0, i - win)
        hi = min(N, i + win + 1)
        if dist[i] <= dist[lo:hi].min():
            contacts.append(i)
    dedup = []
    last = -100
    for c in contacts:
        if c - last > int(min_gap * fps):
            dedup.append(c)
            last = c
    return dedup
