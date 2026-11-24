"""Discriminators and perceptual loss: spectral-norm PatchGAN (LaMa fine-tuning critic and the M4 realness signal),
used also as the Pix2Pix critic in colour de-fading."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .segment import _positions


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
