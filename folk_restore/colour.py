"""Milestone 3: colour de-fading (Reinhard baseline and learned LAB-residual U-Net)."""
import cv2
import numpy as np

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

