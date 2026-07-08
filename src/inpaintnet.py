import numpy as np
import torch
import torch.nn as nn


class _Conv1DBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=3, padding="same", bias=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class _Double1DConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv_1 = _Conv1DBlock(in_dim, out_dim)
        self.conv_2 = _Conv1DBlock(out_dim, out_dim)

    def forward(self, x):
        return self.conv_2(self.conv_1(x))


class InpaintNet(nn.Module):
    """TrackNetV3 trajectory rectification network.

    Inputs:  coor (N, L, 2) normalized coords in [0, 1],
             mask (N, L, 1) with 1 = frame to repair, 0 = keep.
    Outputs: repaired coor (N, L, 2) in [0, 1].
    Only masked frames are replaced: out = pred*mask + coor*(1-mask).
    """

    def __init__(self):
        super().__init__()
        self.down_1 = _Conv1DBlock(3, 32)
        self.down_2 = _Conv1DBlock(32, 64)
        self.down_3 = _Conv1DBlock(64, 128)
        self.buttelneck = _Double1DConv(128, 256)
        self.up_1 = _Conv1DBlock(384, 128)
        self.up_2 = _Conv1DBlock(192, 64)
        self.up_3 = _Conv1DBlock(96, 32)
        self.predictor = nn.Conv1d(32, 2, 3, padding="same")
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, m):
        x = torch.cat([x, m], dim=2)               # (N, L, 3)
        x = x.permute(0, 2, 1)                      # (N, 3, L)
        x1 = self.down_1(x)                         # (N, 32, L)
        x2 = self.down_2(x1)                        # (N, 64, L)
        x3 = self.down_3(x2)                        # (N, 128, L)
        x = self.buttelneck(x3)                    # (N, 256, L)
        x = torch.cat([x, x3], dim=1)               # (N, 384, L)
        x = self.up_1(x)                            # (N, 128, L)
        x = torch.cat([x, x2], dim=1)               # (N, 192, L)
        x = self.up_2(x)                            # (N, 64, L)
        x = torch.cat([x, x1], dim=1)               # (N, 96, L)
        x = self.up_3(x)                            # (N, 32, L)
        x = self.predictor(x)                       # (N, 2, L)
        x = self.sigmoid(x)                         # (N, 2, L)
        x = x.permute(0, 2, 1)                      # (N, L, 2)
        return x


def load_inpaintnet(path, device="cpu"):
    if path is None:
        return None
    try:
        sd = torch.load(path, map_location=device)
    except Exception as e:
        print("WARNING: could not load InpaintNet weights (%s): %s" % (path, e))
        return None
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if not isinstance(sd, dict):
        print("WARNING: InpaintNet weights not a state_dict; skipping rectification.")
        return None
    # Strip DataParallel prefix and normalize the 'buttelneck'/'buttleneck' token.
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    sd = {k.replace("buttleneck", "buttelneck"): v for k, v in sd.items()}
    model = InpaintNet().to(device)
    cur = model.state_dict()
    matched = set(sd.keys()) & set(cur.keys())
    missing = set(cur.keys()) - set(sd.keys())
    extra = set(sd.keys()) - set(cur.keys())
    if missing or extra:
        print(f"InpaintNet: WARNING partial match {len(matched)}/{len(cur)} keys; "
              f"missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}")
        model.load_state_dict(sd, strict=False)
    else:
        model.load_state_dict(sd, strict=True)
        print(f"InpaintNet: loaded {len(matched)}/{len(cur)} param tensors from {path}")
    model.eval()
    return model


def _fill_nan(a):
    a = a.copy()
    n = len(a)
    if n == 0:
        return a
    for c in range(a.shape[1]):
        col = a[:, c]
        good = ~np.isnan(col)
        if good.sum() == 0:
            col[:] = 0.0
        elif good.sum() < n:
            idx = np.where(good)[0]
            col[~good] = np.interp(np.where(~good)[0], idx, col[good])
        a[:, c] = col
    return a


def rectify_trajectory(coords_px, W, H, model, device="cpu", seq_len=16, sigma=None):
    """Repair missing shuttle detections using InpaintNet.

    coords_px: (N, 2) pixel coords, np.nan where TrackNet missed.
    W, H:      original frame size (coords normalized to [0, 1] for the model).
    Returns repaired (N, 2) pixel coords. Only genuine misses are replaced;
    detected points are always kept (repaired = coor*(1-mask)).
    """
    coords = np.array(coords_px, dtype=np.float64)
    N = len(coords)
    if model is None or N == 0:
        return coords

    norm = coords.copy()
    valid = ~np.isnan(norm).any(axis=1)
    norm[:, 0] = np.clip(norm[:, 0] / max(W, 1), 0.0, 1.0)
    norm[:, 1] = np.clip(norm[:, 1] / max(H, 1), 0.0, 1.0)
    norm[~valid] = np.nan
    # Interpolate gaps so the model sees a continuous trajectory as input.
    filled = _fill_nan(norm)
    mask = ~valid

    L = min(seq_len, N)
    if sigma is None:
        sigma = L / 2.0
    center = (L - 1) / 2.0
    w = np.exp(-((np.arange(L) - center) ** 2) / (2 * sigma ** 2))

    out = np.zeros((N, 2))
    wsum = np.zeros(N)
    with torch.no_grad():
        for s in range(0, N - L + 1):
            win = filled[s:s + L]                 # (L, 2)
            m = mask[s:s + L].astype(np.float64).reshape(L, 1)
            tin = torch.tensor(win, dtype=torch.float32).unsqueeze(0).to(device)   # (1, L, 2)
            tm = torch.tensor(m, dtype=torch.float32).unsqueeze(0).to(device)      # (1, L, 1)
            pred = model(tin, tm)[0].cpu().numpy()                                 # (L, 2)
            repaired = pred * m + win * (1.0 - m)                                  # (L, 2)
            for k in range(L):
                out[s + k] += w[k] * repaired[k]
                wsum[s + k] += w[k]

    res = norm.copy()
    for i in range(N):
        if wsum[i] > 0:
            res[i] = out[i] / wsum[i]
    res[:, 0] = np.clip(res[:, 0], 0.0, 1.0) * W
    res[:, 1] = np.clip(res[:, 1], 0.0, 1.0) * H
    return res
