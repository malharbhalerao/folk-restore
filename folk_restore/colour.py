"""Milestone 3: colour de-fading (Reinhard baseline and learned LAB-residual U-Net)."""
import cv2
import numpy as np
import torch

from .segment import _positions


def lab_stats(rgb):
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    return lab.mean(0), lab.std(0)


def reinhard(rgb, ref_mean, ref_std, strength=1.0):
    """Reinhard colour transfer in LAB toward reference statistics (a de-fading baseline)."""
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    m, s = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + 1e-6
    out = (lab - m) / s * ref_std + ref_mean
    out = lab * (1 - strength) + out * strength
    out = cv2.cvtColor(out.astype(np.float32), cv2.COLOR_LAB2RGB)
    return (np.clip(out, 0, 1) * 255).round().astype(np.uint8)


# ----------------------------------------------------------------------------- learned de-fader (step 8)
DEFADE_NORM = np.array([100.0, 80.0, 80.0], np.float32)    # LAB -> roughly [-1, 1] network input
DEFADE_SCALE = np.array([20.0, 30.0, 30.0], np.float32)    # network output (unit-ish) -> LAB residual


def build_defade_model(pretrained=True, encoder="resnet34"):
    """U-Net that predicts a LAB colour residual. The head starts at zero, so the untrained model is the identity."""
    import segmentation_models_pytorch as smp
    m = smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=3)
    head = m.segmentation_head[0]
    head.weight.data.zero_()
    head.bias.data.zero_()
    return m


def defade_forward(model, lab_t):
    """lab_t: Bx3xHxW tensor of raw LAB values -> predicted LAB residual (same shape)."""
    import torch
    norm = torch.tensor(DEFADE_NORM, device=lab_t.device).view(1, 3, 1, 1)
    scale = torch.tensor(DEFADE_SCALE, device=lab_t.device).view(1, 3, 1, 1)
    return model(lab_t / norm) * scale


def load_defade(path, device):
    import torch
    ck = torch.load(path, map_location=device)
    m = build_defade_model(pretrained=False, encoder=ck.get("encoder", "resnet34"))
    m.load_state_dict(ck["model"])
    return m.to(device).eval()


def defade_image(model, rgb, device, strength=1.0, tile=512, stride=384, bs=4):
    """Sliding-window colour restoration. strength blends between the input (0) and the full correction (1)."""
    import torch
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    H, W = lab.shape[:2]
    ph, pw = max(0, tile - H), max(0, tile - W)
    labp = cv2.copyMakeBorder(lab, 0, ph, 0, pw, cv2.BORDER_REPLICATE) if (ph or pw) else lab
    Hp, Wp = labp.shape[:2]
    acc, cnt = np.zeros((Hp, Wp, 3), np.float32), np.zeros((Hp, Wp), np.float32)
    boxes = [(y, x) for y in _positions(Hp, tile, stride) for x in _positions(Wp, tile, stride)]
    use_amp = str(device).startswith("cuda")
    with torch.no_grad():
        for i in range(0, len(boxes), bs):
            chunk = boxes[i:i + bs]
            batch = np.stack([labp[y:y + tile, x:x + tile].transpose(2, 0, 1) for y, x in chunk])
            with torch.autocast(device_type="cuda", enabled=use_amp):
                d = defade_forward(model, torch.from_numpy(batch).to(device)).float().cpu().numpy()
            for (y, x), di in zip(chunk, d):
                acc[y:y + tile, x:x + tile] += di.transpose(1, 2, 0)
                cnt[y:y + tile, x:x + tile] += 1
    delta = (acc / np.maximum(cnt, 1)[..., None])[:H, :W]
    out = cv2.cvtColor((lab + strength * delta).astype(np.float32), cv2.COLOR_LAB2RGB)
    return (np.clip(out, 0, 1) * 255).round().astype(np.uint8)


# ----------------------------------------------------------------------------- Pix2Pix de-fader with protected-margin reference
REF_CH = 7


def lab_of(rgb):
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def ref_stats(rgb_region):
    """mean+std in LAB of an un-faded region (HxWx3 uint8 RGB) -> float32[6]."""
    lab = lab_of(rgb_region).reshape(-1, 3)
    return np.concatenate([lab.mean(0), lab.std(0)]).astype(np.float32)


def ref_vector(stats, has_ref):
    """float32[6] raw LAB stats + flag -> normalised float32[7] network input."""
    v = np.empty(REF_CH, np.float32)
    v[:3] = stats[:3] / DEFADE_NORM
    v[3:6] = stats[3:6] / (DEFADE_NORM / 4.0)
    v[6] = 1.0 if has_ref else 0.0
    return v


def build_g(pretrained=True, encoder="resnet34"):
    import segmentation_models_pytorch as smp
    m = smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None, in_channels=3 + REF_CH, classes=3)
    head = m.segmentation_head[0]
    head.weight.data.zero_()
    head.bias.data.zero_()
    return m


def g_forward(model, lab_t, ref_t):
    """lab_t Bx3xHxW raw LAB, ref_t BxREF_CH  ->  predicted LAB residual Bx3xHxW."""
    norm = torch.tensor(DEFADE_NORM, device=lab_t.device).view(1, 3, 1, 1)
    scale = torch.tensor(DEFADE_SCALE, device=lab_t.device).view(1, 3, 1, 1)
    maps = ref_t[:, :, None, None].expand(-1, -1, lab_t.shape[2], lab_t.shape[3])
    return model(torch.cat([lab_t / norm, maps], 1)) * scale


def d_input(faded_lab, cand_lab):
    norm = torch.tensor(DEFADE_NORM, device=faded_lab.device).view(1, 3, 1, 1)
    return torch.cat([faded_lab / norm, cand_lab / norm], 1)


def load_g(path, device):
    ck = torch.load(path, map_location=device)
    m = build_g(pretrained=False, encoder=ck.get("encoder", "resnet34"))
    m.load_state_dict(ck["model"])
    return m.to(device).eval()


def defade_ref_image(model, rgb, device, margin_stats=None, strength=1.0, tile=512, stride=384, bs=4):
    """Sliding-window colour restoration. margin_stats: float32[6] from ref_stats(), or None for "no reference"."""
    lab = lab_of(rgb)
    H, W = lab.shape[:2]
    has = margin_stats is not None
    stats = margin_stats if has else np.concatenate([lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0)])
    ref = torch.from_numpy(ref_vector(stats, has))[None].to(device)
    ph, pw = max(0, tile - H), max(0, tile - W)
    labp = cv2.copyMakeBorder(lab, 0, ph, 0, pw, cv2.BORDER_REPLICATE) if (ph or pw) else lab
    Hp, Wp = labp.shape[:2]
    acc, cnt = np.zeros((Hp, Wp, 3), np.float32), np.zeros((Hp, Wp), np.float32)
    boxes = [(y, x) for y in _positions(Hp, tile, stride) for x in _positions(Wp, tile, stride)]
    use_amp = str(device).startswith("cuda")
    with torch.no_grad():
        for i in range(0, len(boxes), bs):
            chunk = boxes[i:i + bs]
            batch = np.stack([labp[y:y + tile, x:x + tile].transpose(2, 0, 1) for y, x in chunk])
            with torch.autocast(device_type="cuda", enabled=use_amp):
                d = g_forward(model, torch.from_numpy(batch).to(device), ref.expand(len(chunk), -1)).float().cpu().numpy()
            for (y, x), di in zip(chunk, d):
                acc[y:y + tile, x:x + tile] += di.transpose(1, 2, 0)
                cnt[y:y + tile, x:x + tile] += 1
    delta = (acc / np.maximum(cnt, 1)[..., None])[:H, :W]
    out = cv2.cvtColor((lab + strength * delta).astype(np.float32), cv2.COLOR_LAB2RGB)
    return (np.clip(out, 0, 1) * 255).round().astype(np.uint8)


def margin_strip(rgb, edge, thick=40):
    """The 'protected margin': a strip of `thick` px along one edge (0 left, 1 right, 2 top, 3 bottom)."""
    return [rgb[:, :thick], rgb[:, -thick:], rgb[:thick], rgb[-thick:]][edge % 4]
