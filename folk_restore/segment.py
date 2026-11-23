"""Milestone 1: damage segmentation (CLAHE + black-hat features, U-Net) and mask post-processing."""
import cv2
import numpy as np

from .config import CLASSES


def features(rgb, mode="rich"):
    """Model input. mode='rgb' -> 3 channels; mode='rich' -> RGB + CLAHE-L + black-hat (5 channels)."""
    x = rgb.astype(np.float32) / 255.0
    if mode == "rgb":
        return x
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L).astype(np.float32) / 255.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    bh = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, k).astype(np.float32) / 255.0
    bh = np.clip(bh * 4.0, 0, 1)
    return np.concatenate([x, clahe[..., None], bh[..., None]], axis=2)


# ----------------------------------------------------------------------------- model
def build_model(mode="rich", encoder="resnet34", pretrained=True):
    import segmentation_models_pytorch as smp
    in_ch = 3 if mode == "rgb" else 5
    return smp.Unet(encoder_name=encoder,
                    encoder_weights="imagenet" if pretrained else None,
                    in_channels=in_ch, classes=len(CLASSES))


def load_model(ckpt_path, device):
    import torch
    ck = torch.load(ckpt_path, map_location=device)
    model = build_model(ck["mode"], ck["encoder"], pretrained=False)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck["mode"]


def _positions(n, tile, stride):
    if n <= tile:
        return [0]
    pos = list(range(0, n - tile + 1, stride))
    if pos[-1] != n - tile:
        pos.append(n - tile)
    return pos


def segment_image(model, rgb, mode, device, tile=512, stride=384, bs=4):
    """Sliding-window inference. Returns per-class probabilities, float32 HxWx3."""
    import torch
    H, W = rgb.shape[:2]
    ph, pw = max(0, tile - H), max(0, tile - W)
    img = cv2.copyMakeBorder(rgb, 0, ph, 0, pw, cv2.BORDER_REPLICATE) if (ph or pw) else rgb
    Hp, Wp = img.shape[:2]
    acc = np.zeros((len(CLASSES), Hp, Wp), np.float32)
    cnt = np.zeros((Hp, Wp), np.float32)
    boxes = [(y, x) for y in _positions(Hp, tile, stride) for x in _positions(Wp, tile, stride)]
    use_amp = str(device).startswith("cuda")
    with torch.no_grad():
        for i in range(0, len(boxes), bs):
            chunk = boxes[i:i + bs]
            batch = np.stack([features(img[y:y + tile, x:x + tile], mode).transpose(2, 0, 1) for y, x in chunk])
            t = torch.from_numpy(batch).to(device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                p = torch.sigmoid(model(t).float()).cpu().numpy()
            for (y, x), pi in zip(chunk, p):
                acc[:, y:y + tile, x:x + tile] += pi
                cnt[y:y + tile, x:x + tile] += 1
    prob = (acc / np.maximum(cnt, 1)).transpose(1, 2, 0)
    return prob[:H, :W]


def infill_mask(prob, thr=0.5, dilate=5, min_area=20, classes=None):
    """Union of the chosen damage classes (default all) -> cleaned, dilated binary mask (uint8 0/255)."""
    idx = [CLASSES.index(c) for c in classes] if classes else list(range(prob.shape[2]))
    m = (prob[..., idx].max(axis=2) > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = ((stats[:, cv2.CC_STAT_AREA][lab] >= min_area) & (lab > 0)).astype(np.uint8)
    if dilate and dilate > 1:
        keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate)))
    return keep * 255


def dilate_mask(mask01, dilate):
    m = (mask01 > 0).astype(np.uint8)
    if dilate and dilate > 1:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate)))
    return m * 255



# ----------------------------------------------------------------------------- metrics
def dice_np(pred, gt, eps=1e-6):
    pred, gt = pred.astype(bool), gt.astype(bool)
    return float((2 * (pred & gt).sum() + eps) / (pred.sum() + gt.sum() + eps))
