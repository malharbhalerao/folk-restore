"""Milestone 2 helpers: fine-tuning Big-LaMa on our tiles, plus the PatchGAN discriminator that is reused in Milestone 4.

Big-LaMa (the `simple-lama-inpainting` TorchScript file) is a 51 M-parameter FFC-ResNet generator. TorchScript modules keep
their parameters and gradients, so we can continue training the pretrained weights on in-domain data (synthetic damage on
our own tiles) instead of training an inpainter from scratch on ~6k tiles.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- generator wrapper
def _prep(rgb, mask255, device, mod=8):
    img = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    m = (mask255 > 0).astype(np.float32)[None]
    h, w = img.shape[1:]
    ph, pw = (-h) % mod, (-w) % mod
    if ph or pw:
        img = np.pad(img, ((0, 0), (0, ph), (0, pw)), mode="symmetric")
        m = np.pad(m, ((0, 0), (0, ph), (0, pw)), mode="symmetric")
    return torch.from_numpy(img)[None].to(device), torch.from_numpy((m > 0).astype(np.float32))[None].to(device), h, w


class LamaJit:
    """Big-LaMa from a TorchScript file. path=None -> the original pretrained weights (zero-shot)."""

    def __init__(self, path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_lama_jit(path, self.device).eval()

    def __call__(self, rgb, mask255):
        img, m, h, w = _prep(rgb, mask255, self.device)
        with torch.inference_mode():
            out = self.model(img, m)
        out = out[0].permute(1, 2, 0).float().cpu().numpy()
        return np.clip(out * 255, 0, 255).astype(np.uint8)[:h, :w]


def load_lama_jit(path=None, device="cpu"):
    if path:
        return torch.jit.load(str(path), map_location=device).to(device)
    from simple_lama_inpainting import SimpleLama
    return SimpleLama(device=torch.device(device)).model


# ----------------------------------------------------------------------------- discriminator (also the M4 uncertainty signal)
def _sn(m):
    return nn.utils.spectral_norm(m)


class PatchD(nn.Module):
    """Spectral-norm PatchGAN. Output is a map of "real-ness" logits at 1/8 resolution, receptive field ~62 px.

    Used twice: (1) as the adversarial critic when fine-tuning LaMa, (2) at inference time, where low scores on a
    repaired region mean "this patch does not look like real painting" = the discriminator-based uncertainty signal.
    """

    def __init__(self, in_ch=3, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            _sn(nn.Conv2d(in_ch, ch, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            _sn(nn.Conv2d(ch, ch * 2, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            _sn(nn.Conv2d(ch * 2, ch * 4, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            _sn(nn.Conv2d(ch * 4, ch * 4, 4, 1, 2)), nn.LeakyReLU(0.2, True),
            _sn(nn.Conv2d(ch * 4, 1, 3, 1, 1)))

    def forward(self, x):
        return self.net(x)


def save_disc(path, d, calib=None):
    torch.save({"model": d.state_dict(), "calib": calib}, path)


def load_disc(path, device):
    ck = torch.load(path, map_location=device)
    d = PatchD().to(device)
    d.load_state_dict(ck["model"])
    return d.eval(), ck.get("calib")


@torch.no_grad()
def disc_logits(d, rgb, device, tile=512, stride=384):
    """Sliding-window discriminator logits for a whole image -> float32 HxW map (upsampled from 1/8 resolution)."""
    H, W = rgb.shape[:2]
    x = torch.from_numpy(rgb.astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None].to(device)
    if H <= tile and W <= tile:
        z = d(x)
        return F.interpolate(z, size=(H, W), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    acc, cnt = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    from common import _positions
    for y in _positions(H, tile, stride):
        for xx in _positions(W, tile, stride):
            patch = x[:, :, y:y + tile, xx:xx + tile]
            z = F.interpolate(d(patch), size=patch.shape[2:], mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
            acc[y:y + tile, xx:xx + tile] += z
            cnt[y:y + tile, xx:xx + tile] += 1
    return acc / np.maximum(cnt, 1)


# ----------------------------------------------------------------------------- perceptual loss
class VGGPerceptual(nn.Module):
    """L1 between VGG-16 features (relu2_2, relu3_3, relu4_3) of prediction and target. Needs the ImageNet weights."""

    def __init__(self):
        super().__init__()
        import torchvision
        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features.eval()
        self.s1, self.s2, self.s3 = vgg[:9], vgg[9:16], vgg[16:23]
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        x, y = (x - self.mean) / self.std, (y - self.mean) / self.std
        loss = 0.0
        for s in (self.s1, self.s2, self.s3):
            x, y = s(x), s(y)
            loss = loss + F.l1_loss(x, y)
        return loss


# ----------------------------------------------------------------------------- training masks
def random_blob_mask(h, w, rng, n=None):
    """Irregular blobs and wide brush strokes (LaMa-style 'large masks'), so the model also learns big lacunas."""
    m = np.zeros((h, w), np.uint8)
    for _ in range(n or int(rng.integers(1, 3))):
        if rng.random() < 0.5:
            c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            pts = []
            r0 = rng.uniform(0.05, 0.22) * min(h, w)
            for a in np.linspace(0, 2 * np.pi, 14, endpoint=False):
                r = r0 * rng.uniform(0.5, 1.3)
                pts.append((c[0] + r * np.cos(a), c[1] + r * np.sin(a)))
            cv2.fillPoly(m, [np.array(pts, np.int32)], 255)
        else:
            x, y, ang = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0, 2 * np.pi)
            for _ in range(int(rng.integers(3, 8))):
                nx, ny = x + rng.uniform(20, 70) * np.cos(ang), y + rng.uniform(20, 70) * np.sin(ang)
                cv2.line(m, (int(x), int(y)), (int(nx), int(ny)), 255, int(rng.integers(6, 22)))
                x, y, ang = nx, ny, ang + rng.normal(0, 0.6)
    return (m > 0).astype(np.uint8)


def hole_fraction(mask):
    return float((mask > 0).mean())
