import cv2
import numpy as np
import torch
import torch.nn as nn


class TrackNet(nn.Module):
    def __init__(self, in_channels=3, feat=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, feat, 3, padding=1), nn.BatchNorm2d(feat), nn.ReLU(True),
            nn.Conv2d(feat, feat, 3, padding=1), nn.BatchNorm2d(feat), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat, feat * 2, 3, padding=1), nn.BatchNorm2d(feat * 2), nn.ReLU(True),
            nn.Conv2d(feat * 2, feat * 2, 3, padding=1), nn.BatchNorm2d(feat * 2), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat * 2, feat * 2, 3, padding=1), nn.BatchNorm2d(feat * 2), nn.ReLU(True),
            nn.Conv2d(feat * 2, feat * 2, 3, padding=1), nn.BatchNorm2d(feat * 2), nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(feat * 2, feat * 2, 2, stride=2),
            nn.Conv2d(feat * 2, feat * 2, 3, padding=1), nn.BatchNorm2d(feat * 2), nn.ReLU(True),
            nn.ConvTranspose2d(feat * 2, feat, 2, stride=2),
            nn.Conv2d(feat, feat, 3, padding=1), nn.BatchNorm2d(feat), nn.ReLU(True),
            nn.ConvTranspose2d(feat, feat, 2, stride=2),
            nn.Conv2d(feat, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


class TrackNetShuttle:
    def __init__(self, model_path=None, device="cuda", vis_thresh=0.15, img_size=(288, 512)):
        self.device = device
        self.vis_thresh = vis_thresh
        self.img_size = img_size
        self._buf = []
        self.model = TrackNet(in_channels=3)
        if model_path is not None:
            ckpt = torch.load(model_path, map_location="cpu")
            sd = ckpt.get("model" if isinstance(ckpt, dict) else "state_dict", ckpt)
            if isinstance(sd, dict) and any(k.startswith("module.") for k in sd):
                sd = {k.replace("module.", ""): v for k, v in sd.items()}
            model_keys = set(self.model.state_dict().keys())
            sd_keys = set(sd.keys()) if isinstance(sd, dict) else set()
            matched = model_keys & sd_keys
            print(f"TrackNet: loaded {len(matched)}/{len(model_keys)} param tensors "
                  f"from {model_path}")
            if len(matched) == 0:
                print("WARNING: TrackNet checkpoint keys did NOT match the model architecture "
                      "-> model is UNTRAINED (random); shuttle detection will be wrong.")
                print(f"  checkpoint top-level type: {type(ckpt).__name__}; "
                      f"sd type: {type(sd).__name__}")
                print(f"  checkpoint keys (first 40): {list(sd.keys())[:40]}")
            self.model.load_state_dict(sd, strict=False)
        else:
            print("WARNING: TrackNet model_path is None -> using an UNTRAINED random model.")
        self.model.to(device).eval()

    def _preprocess(self, frames):
        grays = []
        for f in frames:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, (self.img_size[1], self.img_size[0]))
            grays.append(g.astype(np.float32) / 255.0)
        while len(grays) < 3:
            grays.insert(0, grays[0])
        stack = np.stack(grays[-3:], axis=0)
        return torch.from_numpy(stack).unsqueeze(0).to(self.device)

    def predict_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        buf, coords = [], []
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            buf.append(frame)
            if len(buf) >= 3:
                with torch.no_grad():
                    heat = self.model(self._preprocess(buf[-3:]))[0, 0].cpu().numpy()
                mask = heat > self.vis_thresh
                if mask.sum() > 0:
                    ys, xs = np.where(mask)
                    cx = float(xs.mean()) / self.img_size[1] * (buf[-1].shape[1])
                    cy = float(ys.mean()) / self.img_size[0] * (buf[-1].shape[0])
                    coords.append([cx, cy])
                else:
                    coords.append([np.nan, np.nan])
            i += 1
        cap.release()
        return np.array(coords, dtype=np.float64)

    def predict_frames(self, frames):
        coords = []
        for f in frames:
            self._buf.append(f)
            if len(self._buf) >= 3:
                with torch.no_grad():
                    heat = self.model(self._preprocess(self._buf[-3:]))[0, 0].cpu().numpy()
                mask = heat > self.vis_thresh
                if mask.sum() > 0:
                    ys, xs = np.where(mask)
                    cx = float(xs.mean()) / self.img_size[1] * (self._buf[-1].shape[1])
                    cy = float(ys.mean()) / self.img_size[0] * (self._buf[-1].shape[0])
                    coords.append([cx, cy])
                else:
                    coords.append([np.nan, np.nan])
        while len(coords) < len(frames):
            coords.append([np.nan, np.nan])
        return np.array(coords, dtype=np.float64)
