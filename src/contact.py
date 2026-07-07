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
