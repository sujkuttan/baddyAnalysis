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
    def __init__(self, concat_ch, out_c, n_convs):
        super().__init__()
        setattr(self, "conv_1", _ConvBlock(concat_ch, out_c, stride=2, transpose=True))
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

    def __init__(self, in_channels=27, out_channels=8):
        super().__init__()
        self.down_block_1 = _DownBlock(in_channels, 64, 2, downsample=True)
        self.down_block_2 = _DownBlock(64, 128, 2, downsample=True)
        self.down_block_3 = _DownBlock(128, 256, 3, downsample=True)
        self.bottleneck = _DownBlock(256, 512, 3, downsample=False)
        # U-Net skip connections: each up block concatenates the upsampled
        # previous decoder map with the corresponding encoder feature map.
        self.up_block_1 = _UpBlock(512 + 256, 256, 3)   # bottleneck(512) + e3(256)
        self.up_block_2 = _UpBlock(256 + 128, 128, 2)   # up1(256) + e2(128)
        self.up_block_3 = _UpBlock(128 + 64, 64, 2)     # up2(128) + e1(64)
        self.predictor = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.down_block_1(x)
        e2 = self.down_block_2(e1)
        e3 = self.down_block_3(e2)
        b = self.bottleneck(e3)
        up1 = self.up_block_1(torch.cat([b, e3], dim=1))
        up2 = self.up_block_2(torch.cat([up1, e2], dim=1))
        up3 = self.up_block_3(torch.cat([up2, e1], dim=1))
        return torch.sigmoid(self.predictor(up3))


class TrackNetShuttle:
    def __init__(self, model_path=None, device="cuda", vis_thresh=0.15, img_size=(288, 512)):
        self.device = device
        self.vis_thresh = vis_thresh
        self.img_size = img_size
        self._buf = []
        self.model = TrackNet(in_channels=27)
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
        # TrackNetV3: 9 consecutive RGB frames -> 27-channel input (9*3).
        arr = []
        for f in frames[-9:]:
            im = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            arr.append(im)
        while len(arr) < 9:
            arr.insert(0, arr[0])
        stack = np.stack(arr, axis=0)            # (9, H, W, 3)
        stack = stack.transpose(0, 3, 1, 2)      # (9, 3, H, W)
        stack = stack.reshape(27, self.img_size[0], self.img_size[1])
        return torch.from_numpy(stack).unsqueeze(0).to(self.device)

    def predict_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        buf, coords = [], []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            buf.append(frame)
            if len(buf) >= 9:
                win = buf[-9:]
                with torch.no_grad():
                    heat = self.model(self._preprocess(win))[0].cpu().numpy()
                Hh, Ww = self.img_size
                fw = buf[-1].shape[1]
                fh = buf[-1].shape[0]
                best = None
                best_mass = 0.0
                for k in range(heat.shape[0]):
                    mask = heat[k] > self.vis_thresh
                    mass = float(mask.sum())
                    if mass > best_mass:
                        best_mass = mass
                        ys, xs = np.where(mask)
                        best = (float(xs.mean()) / Ww * fw, float(ys.mean()) / Hh * fh)
                coords.append(list(best) if best is not None else [np.nan, np.nan])
            else:
                coords.append([np.nan, np.nan])
        cap.release()
        return np.array(coords, dtype=np.float64)

    def predict_frames(self, frames):
        coords = []
        for f in frames:
            self._buf.append(f)
            if len(self._buf) >= 9:
                win = self._buf[-9:]
                with torch.no_grad():
                    heat = self.model(self._preprocess(win))[0].cpu().numpy()  # (8, H, W)
                Hh, Ww = self.img_size
                fw = self._buf[-1].shape[1]
                fh = self._buf[-1].shape[0]
                best = None
                best_mass = 0.0
                for k in range(heat.shape[0]):
                    mask = heat[k] > self.vis_thresh
                    mass = float(mask.sum())
                    if mass > best_mass:
                        best_mass = mass
                        ys, xs = np.where(mask)
                        best = (float(xs.mean()) / Ww * fw, float(ys.mean()) / Hh * fh)
                coords.append(list(best) if best is not None else [np.nan, np.nan])
            else:
                coords.append([np.nan, np.nan])
        while len(coords) < len(frames):
            coords.append([np.nan, np.nan])
        return np.array(coords, dtype=np.float64)
