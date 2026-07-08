import cv2
import numpy as np
import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, transpose=False):
        super().__init__()
        if transpose:
            self.conv = nn.ConvTranspose2d(in_c, out_c, 3, stride=stride,
                                           padding=1, output_padding=1 if stride > 1 else 0,
                                           bias=False)
        else:
            self.conv = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)))


class _DownBlock(nn.Module):
    def __init__(self, in_c, out_c, n_convs, downsample=True):
        super().__init__()
        setattr(self, "conv_1", _ConvBlock(in_c, out_c, stride=2 if downsample else 1))
        for i in range(2, n_convs + 1):
            setattr(self, f"conv_{i}", _ConvBlock(out_c, out_c, stride=1))

    def forward(self, x):
        for i in range(1, len(self._modules) + 1):
            x = getattr(self, f"conv_{i}")(x)
        return x


class _UpBlock(nn.Module):
    def __init__(self, in_c, out_c, n_convs):
        super().__init__()
        setattr(self, "conv_1", _ConvBlock(in_c, out_c, stride=2, transpose=True))
        for i in range(2, n_convs + 1):
            setattr(self, f"conv_{i}", _ConvBlock(out_c, out_c, stride=1))

    def forward(self, x):
        for i in range(1, len(self._modules) + 1):
            x = getattr(self, f"conv_{i}")(x)
        return x


class TrackNet(nn.Module):
    """Canonical TrackNet: down_block_1..3 -> bottleneck -> up_block_1..3 -> predictor.

    Matches the published checkpoint key naming (down_block_1.conv_1.conv.weight,
    up_block_1.conv_1.bn.*, predictor.weight, ...). Downsampling is via stride-2
    convs, upsampling via stride-2 transpose-convs; output heatmap is sigmoided.
    """

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.down_block_1 = _DownBlock(in_channels, 64, 2, downsample=True)
        self.down_block_2 = _DownBlock(64, 128, 2, downsample=True)
        self.down_block_3 = _DownBlock(128, 256, 3, downsample=True)
        self.bottleneck = _DownBlock(256, 512, 3, downsample=False)
        self.up_block_1 = _UpBlock(512, 256, 3)
        self.up_block_2 = _UpBlock(256, 128, 2)
        self.up_block_3 = _UpBlock(128, 64, 2)
        self.predictor = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x = self.down_block_1(x)
        x = self.down_block_2(x)
        x = self.down_block_3(x)
        x = self.bottleneck(x)
        x = self.up_block_1(x)
        x = self.up_block_2(x)
        x = self.up_block_3(x)
        return torch.sigmoid(self.predictor(x))


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
            self.model.load_state_dict(sd, strict=True)
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
