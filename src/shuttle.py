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
        self.n_convs = n_convs
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        setattr(self, "conv_1", _ConvBlock(concat_ch, out_c, stride=1))
        for i in range(2, n_convs + 1):
            setattr(self, f"conv_{i}", _ConvBlock(out_c, out_c, stride=1))

    def forward(self, x):
        x = self.upsample(x)
        for i in range(1, self.n_convs + 1):
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


def _predict_location(heatmap):
    """Reference `test.py`: largest connected-component bbox center.

    heatmap: single-channel (H, W) float/uint8 (already thresholded >0.5).
    Returns (cx, cy) in input-pixel space, or None if empty.
    """
    if heatmap is None or np.amax(heatmap) == 0:
        return None
    h = (heatmap > 0.5).astype(np.uint8) if heatmap.dtype != np.uint8 else heatmap
    cnts, _ = cv2.findContours(h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    rects = [cv2.boundingRect(c) for c in cnts]
    x, y, w, h_ = max(rects, key=lambda r: r[2] * r[3])
    return (x + w / 2.0, y + h_ / 2.0)


class TrackNetShuttle:
    """TrackNetV3 trajectory predictor (official weights: seq_len=8, bg_mode='concat').

    Input is 27 channels = [background, f0, ..., f7] where background is the
    per-video median frame (concat mode). Decoding follows predict.py:
    sliding-window ensemble (step=1) of raw sigmoids, threshold >0.5, then the
    largest connected-component centroid per ensembled heatmap.
    """

    def __init__(self, model_path=None, device="cuda", img_size=(288, 512), seq_len=8):
        self.device = device
        self.img_size = img_size
        self.seq_len = seq_len
        self._buf = []
        self.model = TrackNet(in_channels=(seq_len + 1) * 3, out_channels=seq_len)
        if model_path is not None:
            ckpt = torch.load(model_path, map_location="cpu")
            sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
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
            pd = ckpt.get("param_dict") if isinstance(ckpt, dict) else None
            if pd and pd.get("seq_len") and pd["seq_len"] != seq_len:
                print(f"WARNING: checkpoint trained with seq_len={pd['seq_len']} "
                      f"but model built with seq_len={seq_len}; using checkpoint value.")
                self.seq_len = int(pd["seq_len"])
                self.model = TrackNet(in_channels=(self.seq_len + 1) * 3,
                                      out_channels=self.seq_len)
                self.model.load_state_dict(sd, strict=True)
        else:
            print("WARNING: TrackNet model_path is None -> using an UNTRAINED random model.")
        self.model.to(device).eval()
        # Triangular ensemble weights (reference get_ensemble_weight, mode='weight').
        s = self.seq_len
        w = np.ones(s, dtype=np.float64)
        for i in range(int(np.ceil(s / 2))):
            w[i] = i + 1
            w[s - i - 1] = i + 1
        self.weight = w / w.sum()

    def _build_input(self, window_small, bg):
        """window_small: list of seq_len already-resized (Hh,Ww,3) float32 frames;
        bg: median (Hh,Ww,3) uint8 frame. Returns (1, (seq_len+1)*3, H, W) tensor in
        [0,1], concat=[bg, f0..f7]. Frames are expected pre-resized (RAM guard)."""
        H, W = self.img_size
        chans = []
        bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chans.append(bg_rgb.transpose(2, 0, 1))
        for f in window_small:
            im = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            chans.append(im.transpose(2, 0, 1))            # (3, H, W)
        stack = np.concatenate(chans, axis=0)              # ((seq_len+1)*3, H, W)
        return torch.from_numpy(stack).unsqueeze(0).to(self.device)

    def predict_video(self, video_path, max_frames=None):
        """Stream the video, keeping only resized frames in RAM (RAM guard), so a
        full-length clip never holds native-resolution frames. Returns (N,2) coords."""
        cap = cv2.VideoCapture(video_path)
        Hh, Ww = self.img_size
        small = []
        sum_arr = None
        n = 0
        while True:
            ret, f = cap.read()
            if not ret:
                break
            im = cv2.resize(f, (Ww, Hh)).astype(np.float32)
            small.append(im)
            sum_arr = im.astype(np.float64) if sum_arr is None else sum_arr + im
            n += 1
            if max_frames is not None and n >= max_frames:
                break
        cap.release()
        if n == 0:
            return np.array([], dtype=np.float64)
        bg = (sum_arr / n).astype(np.uint8)        # mean = median proxy, cheap + RAM-safe
        return self._predict_from_small(small, bg, fw=f.shape[1], fh=f.shape[0])

    def _predict_from_small(self, small, bg, fw, fh):
        """Like _predict_list but operates on already-resized frames (RAM guard)."""
        seq_len = self.seq_len
        n = len(small)
        if n == 0:
            return np.array([], dtype=np.float64)
        Hh, Ww = self.img_size
        accum = np.zeros((n, Hh, Ww), dtype=np.float64)
        wsum = np.zeros(n, dtype=np.float64)
        w = self.weight
        chunk = 8

        def _add_heat(win_start, heat):
            if win_start + seq_len <= n:
                for j in range(seq_len):
                    fidx = win_start + j
                    accum[fidx] += w[j] * heat[j]
                    wsum[fidx] += w[j]
            else:
                f = n - win_start
                for j in range(f):
                    fidx = win_start + j
                    accum[fidx] += heat[j]
                    wsum[fidx] += 1.0

        with torch.no_grad():
            for start in range(0, n, chunk):
                end = min(start + chunk, n - seq_len + 1)
                if end <= start:
                    break
                batch = [self._build_input(small[s:s + seq_len], bg) for s in range(start, end)]
                x = torch.cat(batch, dim=0).to(self.device)
                h = self.model(x).cpu().numpy()
                for k, s in enumerate(range(start, end)):
                    _add_heat(s, h[k])
                del x, h, batch
            for start in range(max(n - seq_len + 1, 0), n):
                win = small[start:start + seq_len]
                win = win + [win[-1]] * (seq_len - len(win))
                x = self._build_input(win, bg).to(self.device)
                h = self.model(x).cpu().numpy()[0]
                _add_heat(start, h)
                del x, h

        coords = []
        for i in range(n):
            if wsum[i] == 0:
                coords.append([np.nan, np.nan])
                continue
            c = _predict_location(accum[i] / wsum[i])
            if c is None:
                coords.append([np.nan, np.nan])
            else:
                coords.append([c[0] / Ww * fw, c[1] / Hh * fh])
        return np.array(coords, dtype=np.float64)

    def predict_frames(self, frames):
        """frames: list of BGR frames (a batch). Buffers across calls like before
        but uses concat-background + sliding-window ensemble decoding."""
        return self._predict_list(list(self._buf) + list(frames), append_buf=True)

    def _predict_list(self, frames, append_buf=False, max_frames=None):
        """RAM-guarded entry used by the pipeline (feeds 128-frame batches)."""
        if max_frames is not None:
            frames = frames[:max_frames]
        if append_buf:
            self._buf = frames
        if len(frames) == 0:
            return np.array([], dtype=np.float64)
        Hh, Ww = self.img_size
        fw, fh = frames[-1].shape[1], frames[-1].shape[0]
        # Resize to model size up front and drop full-res frames (RAM guard).
        small = [cv2.resize(f, (Ww, Hh)).astype(np.float32) for f in frames]
        del frames
        bg = np.median(np.array(small, dtype=np.float64), axis=0).astype(np.uint8)
        return self._predict_from_small(small, bg, fw, fh)
