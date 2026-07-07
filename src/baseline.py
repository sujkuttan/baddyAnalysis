import numpy as np

from .config import COURT_LENGTH, canonical_stroke


def geometry_stroke(foot_court, wrist_court, shuttle_post_dir, player_height_m=1.7):
    if np.any(np.isnan(foot_court)) or np.any(np.isnan(wrist_court)):
        return "clear"
    y = foot_court[1]
    back = y > COURT_LENGTH * 0.6
    front = y < COURT_LENGTH * 0.35
    contact_h = wrist_court[1]
    high = contact_h > (COURT_LENGTH * 0.5)
    dx, dy = shuttle_post_dir[0], shuttle_post_dir[1]
    downward = dy < -0.5
    steep_down = dy < -2.0
    if front and not high:
        return "net_shot"
    if back and high and steep_down:
        return "smash"
    if back and high and downward:
        return "clear"
    if back and not high:
        return "lift"
    if abs(dx) > abs(dy) and abs(dx) > 1.0:
        return "drive"
    if front and high:
        return "push"
    return "drop"


def build_baseline_predictions(contact_frames, attrib, foot_streams, wrist_streams,
                               shuttle_court, fps):
    preds = []
    for cf, pid in zip(contact_frames, attrib):
        if pid is None or pid not in foot_streams:
            preds.append("clear")
            continue
        foot = foot_streams[pid][cf] if cf < len(foot_streams[pid]) else np.array([np.nan, np.nan])
        wrist = wrist_streams.get(pid, [np.array([np.nan, np.nan])] * (cf + 1))[cf] if pid in wrist_streams else np.array([np.nan, np.nan])
        post = shuttle_court[min(cf + 2, len(shuttle_court) - 1)] - shuttle_court[cf]
        dt = 1.0 / fps
        preds.append(geometry_stroke(foot, wrist, post / dt if dt else post))
    return preds
