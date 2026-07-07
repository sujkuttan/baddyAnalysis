import numpy as np
import cv2


class FlowExtractor:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = None
        try:
            import torch
            from torchvision.models.optical_flow import raft_large
            self.model = raft_large(pretrained=True).to(device).eval()
            self.torch = torch
        except Exception:
            self.model = None

    def flow(self, frame_a, frame_b):
        ga = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
        if self.model is not None:
            import torch
            ta = torch.from_numpy(ga).permute(2, 0, 1).float() / 255.0
            tb = torch.from_numpy(gb).permute(2, 0, 1).float() / 255.0
            with torch.no_grad():
                out = self.model(ta.unsqueeze(0).to(self.device), tb.unsqueeze(0).to(self.device))
                f = out[-1][0].permute(1, 2, 0).cpu().numpy()
            return f
        return cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def _mbh(flow):
    fx = flow[:, :, 0]
    fy = flow[:, :, 1]
    mhx = cv2.Sobel(fx, cv2.CV_32F, 1, 0, ksize=3)
    mhy = cv2.Sobel(fy, cv2.CV_32F, 0, 1, ksize=3)
    return mhx, mhy


def mbh_histograms(flow, bbox, grid=(4, 4), nbins=8):
    mhx, mhy = _mbh(flow)
    x0, y0, x1, y1 = [int(v) for v in bbox]
    h, w = mhx.shape
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w - 1, x1), min(h - 1, y1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros(grid[0] * grid[1] * nbins * 2)
    sub_mhx = mhx[y0:y1, x0:x1]
    sub_mhy = mhy[y0:y1, x0:x1]
    mag = np.sqrt(sub_mhx ** 2 + sub_mhy ** 2)
    ang = (np.arctan2(sub_mhy, sub_mhx) * 180 / np.pi) % 360
    gh, gw = grid
    cell_h, cell_w = max(1, (y1 - y0) // gh), max(1, (x1 - x0) // gw)
    feats = []
    for iy in range(gh):
        for ix in range(gw):
            yy0, yy1 = y0 + iy * cell_h, min(y1, y0 + (iy + 1) * cell_h)
            xx0, xx1 = x0 + ix * cell_w, min(x1, x0 + (ix + 1) * cell_w)
            if yy1 <= yy0 or xx1 <= xx0:
                feats.append(np.zeros(nbins * 2))
                continue
            m = mag[yy0:yy1, xx0:xx1].ravel()
            a = ang[yy0:yy1, xx0:xx1].ravel()
            hx, _ = np.histogram(a, bins=nbins, range=(0, 360), weights=m)
            hy, _ = np.histogram(a, bins=nbins, range=(0, 360), weights=m)
            block = np.concatenate([hx, hy])
            norm = np.linalg.norm(block)
            feats.append(block / norm if norm > 0 else block)
    return np.concatenate(feats)


def compute_mbh_sequence(video_path, bboxes_per_frame, grid=(4, 4), nbins=8):
    cap = cv2.VideoCapture(video_path)
    fe = FlowExtractor()
    prev = None
    seq = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if prev is None:
            prev = frame
            seq.append(np.zeros(grid[0] * grid[1] * nbins * 2))
            continue
        flow = fe.flow(prev, frame)
        bf = bboxes_per_frame.pop(0) if bboxes_per_frame else [0, 0, frame.shape[1], frame.shape[0]]
        seq.append(mbh_histograms(flow, bf, grid, nbins))
        prev = frame
    cap.release()
    return np.array(seq, dtype=np.float64)
