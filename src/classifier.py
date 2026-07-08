import numpy as np
import torch
import torch.nn as nn


def normalize_pose_seq(pose_seq_court):
    # pose_seq_court: (T, 17, 2) joint-major, or a single (17, 2) frame
    pose_seq_court = np.array(pose_seq_court, dtype=np.float64)
    if pose_seq_court.ndim == 2:
        pose_seq_court = pose_seq_court[None]
    out = []
    for seq in pose_seq_court:  # seq is (17, 2): rows = joints, cols = x/y
        hip = (seq[11] + seq[12]) / 2.0            # (2,)
        torso = float(np.linalg.norm(seq[5] - seq[11]))
        if torso < 1e-3:
            torso = 1.0
        n = (seq - hip) / torso                     # (17, 2)
        out.append(n.reshape(-1))                   # (34,)
    return np.array(out, dtype=np.float64)          # (T, 34)


class StreamEncoder(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, 3, padding=1), nn.ReLU(True),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU(True),
            nn.AdaptiveMaxPool1d(1),
        )
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.transpose(1, 2)
        h = self.net(x).squeeze(-1)
        return self.proj(h)


class FusionClassifier(nn.Module):
    def __init__(self, dims, n_classes, hidden=128):
        super().__init__()
        self.encoders = nn.ModuleList([StreamEncoder(d, 64, 64) for d in dims])
        self.head = nn.Sequential(
            nn.Linear(64 * len(dims), hidden), nn.ReLU(True),
            nn.Dropout(0.3), nn.Linear(hidden, n_classes),
        )

    def forward(self, streams):
        emb = [e(s) for e, s in zip(self.encoders, streams)]
        return self.head(torch.cat(emb, dim=1))


def _interp_pose(seq):
    seq = seq.copy()
    for j in range(seq.shape[1]):
        for c in range(seq.shape[2]):
            col = seq[:, j, c]
            if np.all(np.isnan(col)):
                col[:] = 0.0
            else:
                idx = np.arange(len(col))
                good = ~np.isnan(col)
                col[~good] = np.interp(idx[~good], idx[good], col[good])
    return seq


def extract_stroke_windows(court_poses, racket_traj, mbh_seq, contact_frames,
                            attrib, window=20, n_joints=17):
    samples = []
    for cf, pid in zip(contact_frames, attrib):
        if pid is None:
            continue
        if pid not in court_poses:
            continue
        lo = max(0, cf - window // 2)
        hi = min(len(court_poses[pid]), cf + window // 2 + 1)
        if hi - lo < 5:
            continue
        pseq = court_poses[pid][lo:hi]
        if np.any(np.isnan(pseq)):
            pseq = _interp_pose(pseq)
        rseq = racket_traj.get(pid, np.zeros((len(court_poses[pid]), 2)))[lo:hi]
        if np.any(np.isnan(rseq)) or len(rseq) == 0:
            rseq = np.zeros((hi - lo, 2))
        mseq = mbh_seq[lo:hi]
        samples.append({
            "pose": normalize_pose_seq(pseq),
            "racket": np.nan_to_num(rseq),
            "mbh": mseq if mseq.ndim == 2 else np.zeros((hi - lo, 1)),
            "contact": cf,
            "player": pid,
        })
    return samples


def _to_tensor(a):
    return torch.from_numpy(np.array(a, dtype=np.float64)).float()


def train_classifier(samples_train, samples_val, n_classes, epochs=40, lr=1e-3, device="cpu"):
    dims = [samples_train[0]["pose"].shape[1],
            samples_train[0]["racket"].shape[1],
            samples_train[0]["mbh"].shape[1]]
    model = FusionClassifier(dims, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        for s in samples_train:
            streams = [_to_tensor(s[k]).unsqueeze(0).to(device) for k in ("pose", "racket", "mbh")]
            y = torch.tensor([s["label"]], device=device)
            opt.zero_grad()
            loss = loss_fn(model(streams), y)
            loss.backward()
            opt.step()
    return model


def predict_classifier(model, samples, device="cpu"):
    model.eval()
    preds = []
    with torch.no_grad():
        for s in samples:
            streams = [_to_tensor(s[k]).unsqueeze(0).to(device) for k in ("pose", "racket", "mbh")]
            logits = model(streams)
            preds.append(int(logits.argmax(1).item()))
    return preds
