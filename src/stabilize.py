import cv2
import numpy as np

from .config import COURT_CORNERS_METERS, COURT_WIDTH, COURT_LENGTH


def boundary_sample_points(n_per_side=6):
    w, l = COURT_WIDTH, COURT_LENGTH
    pts = []
    for i in range(n_per_side):
        t = i / (n_per_side - 1)
        pts.append([t * w, 0.0])
        pts.append([t * w, l])
    for i in range(n_per_side):
        t = i / (n_per_side - 1)
        pts.append([0.0, t * l])
        pts.append([w, t * l])
    return np.array(pts, dtype=np.float64)


def init_stabilizer(corners0, n_per_side=6):
    corners0 = np.array(corners0, dtype=np.float64)
    H0 = cv2.findHomography(corners0, COURT_CORNERS_METERS)[0]
    court_pts = boundary_sample_points(n_per_side)
    img_pts0 = cv2.perspectiveTransform(
        np.expand_dims(court_pts, 0), np.linalg.inv(H0)
    )[0]
    return {
        "H0": H0,
        "court_pts": court_pts,
        "img_pts": img_pts0.astype(np.float32),
    }


def _lk_track(prev_gray, gray, pts):
    if prev_gray is None or pts is None or len(pts) == 0:
        return pts, np.zeros((len(pts), 1), dtype=np.uint8)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, pts.reshape(-1, 1, 2).astype(np.float32), None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    return nxt, st


def stabilize_video(video_path, corners0, n_per_side=6, max_frames=None):
    state = init_stabilizer(corners0, n_per_side)
    court_pts = state["court_pts"]
    img_pts = state["img_pts"].copy()
    H_prev = state["H0"]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = 0
    prev_gray = None
    Hs = []
    used_counts = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and n >= max_frames:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            nxt, st = _lk_track(prev_gray, gray, img_pts)
            good_mask = (st.reshape(-1) == 1)
            nxt_good = nxt.reshape(-1, 2)[good_mask]
            court_good = court_pts[good_mask]
            if len(nxt_good) >= max(6, 0.4 * len(court_pts)):
                H, mask = cv2.findHomography(
                    nxt_good.reshape(-1, 1, 2),
                    court_good.reshape(-1, 1, 2),
                    cv2.RANSAC, 4.0,
                )
                if H is not None:
                    H_prev = H
                    img_pts = nxt_good.astype(np.float32)
                    used_counts.append(int(good_mask.sum()))
                else:
                    used_counts.append(int(good_mask.sum()))
            else:
                used_counts.append(int(good_mask.sum()))
        Hs.append(H_prev.copy())
        prev_gray = gray
        img_pts = img_pts
        n += 1

    cap.release()
    return {"homographies": Hs, "fps": fps, "n_frames": len(Hs), "used_counts": used_counts}


def warp_points(H, pts_image):
    pts = np.array(pts_image, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.shape[1] == 2:
        pts = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    out = (H @ pts.T).T
    out = out[:, :2] / out[:, 2:3]
    return out


def interactive_select_corners(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("cannot read video for corner selection")
    corners = []

    def click(e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append([x, y])
            cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
            cv2.imshow("click 4 court corners: TL, TR, BR, BL", frame)

    cv2.imshow("click 4 court corners: TL, TR, BR, BL", frame)
    cv2.setMouseCallback("click 4 court corners: TL, TR, BR, BL", click)
    while len(corners) < 4:
        cv2.waitKey(50)
    cv2.destroyAllWindows()
    return np.array(corners, dtype=np.float64)
