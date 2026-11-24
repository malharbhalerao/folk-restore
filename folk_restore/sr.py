"""Milestone 4, part 1: Real-ESRGAN x4 super-resolution with memory-safe tiled inference.

The RRDBNet architecture is re-implemented here (about 50 lines) and loads the official RealESRGAN_x4plus.pth weights, so
nothing needs `basicsr` / `realesrgan` or the torchvision patch that the README mentioned.

Tiling: the low-resolution image is cut into overlapping windows, each window is upscaled separately (so GPU memory depends
on the window size, not on the painting size), and the outputs are blended with a linear ramp across the overlap so no seams
show. A larger overlap = fewer seam artefacts but more compute. `seam_report()` in 12_eval_sr_uncertainty.py measures it.
"""
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"


class _RDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.act(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.act(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        return self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) * 0.2 + x


class _RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = _RDB(nf, gc), _RDB(nf, gc), _RDB(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    """RealESRGAN_x4plus generator: 23 RRDB blocks, RGB in [0,1] -> RGB x4."""

    def __init__(self, nf=64, nb=23, gc=32):
        super().__init__()
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
        self.body = nn.Sequential(*[_RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.act(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.act(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.act(self.conv_hr(feat)))


def weights_path(url=WEIGHTS_URL):
    d = Path(torch.hub.get_dir()) / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    p = d / os.path.basename(url)
    if not p.exists():
        print(f"downloading {url} -> {p}")
        urllib.request.urlretrieve(url, p)
    return p


def load_esrgan(device=None, path=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(path or weights_path(), map_location="cpu")
    net = RRDBNet()
    net.load_state_dict(sd.get("params_ema", sd), strict=True)
    return net.eval().to(device)


def _ramp(n, overlap, first, last):
    """1-D blend weights for a window of n output px: linear ramp over `overlap` px on each interior side."""
    w = np.ones(n, np.float32)
    r = max(1, int(overlap))
    ramp = np.linspace(1.0 / r, 1.0, r, dtype=np.float32)
    if not first:
        w[:r] = np.minimum(w[:r], ramp)
    if not last:
        w[-r:] = np.minimum(w[-r:], ramp[::-1])
    return w


def _starts(n, tile, stride):
    if n <= tile:
        return [0]
    pos = list(range(0, n - tile + 1, stride))
    if pos[-1] != n - tile:
        pos.append(n - tile)
    return pos


@torch.no_grad()
def sr_x4(net, rgb, tile=256, overlap=32, device=None, half=None):
    """Tiled x4 super-resolution. rgb: HxWx3 uint8 -> (4H)x(4W)x3 uint8. tile=0 disables tiling (one pass)."""
    device = device or next(net.parameters()).device
    half = (str(device).startswith("cuda")) if half is None else half
    H, W = rgb.shape[:2]

    def run(patch):
        x = torch.from_numpy(patch.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(half)):
            y = net(x)
        return y.float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()

    if not tile or (H <= tile and W <= tile):
        return (run(rgb) * 255).round().astype(np.uint8)
    stride = tile - overlap
    acc = np.zeros((H * 4, W * 4, 3), np.float32)
    wsum = np.zeros((H * 4, W * 4), np.float32)
    ys, xs = _starts(H, tile, stride), _starts(W, tile, stride)
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            th, tw = min(tile, H - y), min(tile, W - x)
            out = run(rgb[y:y + th, x:x + tw])
            wy = _ramp(th * 4, overlap * 4, yi == 0, yi == len(ys) - 1)
            wx = _ramp(tw * 4, overlap * 4, xi == 0, xi == len(xs) - 1)
            w = wy[:, None] * wx[None, :]
            acc[y * 4:(y + th) * 4, x * 4:(x + tw) * 4] += out * w[..., None]
            wsum[y * 4:(y + th) * 4, x * 4:(x + tw) * 4] += w
    return ((acc / np.maximum(wsum, 1e-6)[..., None]) * 255).round().clip(0, 255).astype(np.uint8)


def super_resolve(net, rgb, scale=2, tile=256, overlap=32):
    """Real-ESRGAN at x4, then area-resampled to the requested factor (as the official `outscale` option does)."""
    out = sr_x4(net, rgb, tile, overlap)
    if scale == 4:
        return out
    h, w = rgb.shape[:2]
    return cv2.resize(out, (w * scale, h * scale), interpolation=cv2.INTER_AREA)
