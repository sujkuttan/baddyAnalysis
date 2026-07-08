import numpy as np
from scipy.ndimage import median_filter

from .config import COURT_WIDTH, COURT_LENGTH


def _smooth(pos, win=3):
    """Temporal median filter to suppress pose-jitter spikes; preserves nans."""
    pos = np.array(pos, dtype=np.float64)
    if win < 2 or len(pos) < 2:
        return pos
    out = pos.copy()
    nanmask = np.isnan(pos)
    for c in range(pos.shape[1]):
        col = pos[:, c]
        valid = ~np.isnan(col)
        if valid.sum() < 2:
            continue
        col_f = col.copy()
        col_f[~valid] = np.interp(np.where(~valid)[0], np.where(valid)[0], col[valid])
        sm = median_filter(col_f, size=win, mode="nearest")
        sm[nanmask[:, c]] = np.nan
        out[:, c] = sm
    return out


def _speed(pos, fps, max_step=0.8):
    pos = np.array(pos, dtype=np.float64)
    v = np.full(len(pos), np.nan)
    if len(pos) < 2:
        return v
    dt = 1.0 / fps
    d = np.sqrt(np.sum((pos[1:] - pos[:-1]) ** 2, axis=1))
    step_ok = (~np.any(np.isnan(pos[1:]), axis=1)) & (~np.any(np.isnan(pos[:-1]), axis=1)) & (d <= max_step)
    v[1:] = np.where(step_ok, d / dt, np.nan)
    v[0] = v[1]
    return v


def compute_movement(positions, fps, max_step=0.8):
    out = {}
    for pid, pos in positions.items():
        pos = _smooth(pos, win=3)
        valid = ~np.any(np.isnan(pos), axis=1)
        speed = _speed(pos, fps, max_step=max_step)
        if valid.sum() > 1:
            d = np.sqrt(np.sum((pos[1:] - pos[:-1]) ** 2, axis=1))
            step_ok = (~np.any(np.isnan(pos[1:]), axis=1)) & (~np.any(np.isnan(pos[:-1]), axis=1)) & (d <= max_step)
            total = float(np.nansum(np.where(step_ok, d, 0.0)))
        else:
            total = 0.0
        out[pid] = {
            "positions": pos,
            "valid": valid,
            "speed": speed,
            "total_distance_m": total,
            "mean_speed": float(np.nanmean(speed)) if valid.any() else 0.0,
            "max_speed": float(np.nanmax(speed)) if valid.any() else 0.0,
        }
    return out


def court_heatmap(positions, grid=(15, 30)):
    hm = np.zeros(grid, dtype=np.float64)
    gx, gy = grid
    for pid, pos in positions.items():
        pos = np.array(pos, dtype=np.float64)
        pos = pos[~np.any(np.isnan(pos), axis=1)]
        if len(pos) == 0:
            continue
        ix = np.clip((pos[:, 0] / COURT_WIDTH * (gx - 1)).astype(int), 0, gx - 1)
        iy = np.clip((pos[:, 1] / COURT_LENGTH * (gy - 1)).astype(int), 0, gy - 1)
        for x, y in zip(ix, iy):
            hm[x, y] += 1
    if hm.sum() > 0:
        hm = hm / hm.sum()
    return hm


def zone_coverage(positions, nz=3):
    cov = {}
    for pid, pos in positions.items():
        pos = np.array(pos, dtype=np.float64)
        pos = pos[~np.any(np.isnan(pos), axis=1)]
        zx = np.clip((pos[:, 0] / COURT_WIDTH * nz).astype(int), 0, nz - 1)
        zy = np.clip((pos[:, 1] / COURT_LENGTH * nz).astype(int), 0, nz - 1)
        zones = set(zip(zx.tolist(), zy.tolist()))
        cov[pid] = len(zones) / (nz * nz)
    return cov


def fatigue_profile(positions, fps, window_sec=30.0, max_step=0.8):
    res = {}
    for pid, pos in positions.items():
        pos = _smooth(pos, win=3)
        speed = _speed(pos, fps, max_step=max_step)
        valid = ~np.isnan(speed)
        speed = speed[valid]
        w = max(1, int(window_sec * fps))
        if len(speed) < w:
            res[pid] = {"trend_slope": 0.0, "first_half": float(np.nanmean(speed)),
                        "second_half": float(np.nanmean(speed)), "fatigue_score": 0.0}
            continue
        half = len(speed) // 2
        first = float(np.mean(speed[:half]))
        second = float(np.mean(speed[half:]))
        slope = (second - first) / max(1e-6, first)
        res[pid] = {
            "first_half_mean_speed": first,
            "second_half_mean_speed": second,
            "trend_slope": slope,
            "fatigue_score": float(-slope),
        }
    return res


def recovery_time(contact_frames, fps, idle_speed_thresh=0.6):
    if len(contact_frames) < 2:
        return []
    rec = []
    for a, b in zip(contact_frames[:-1], contact_frames[1:]):
        rec.append((b - a) / fps)
    return rec
