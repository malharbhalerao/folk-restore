"""Shared helpers for the folk-painting restoration pipeline."""
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASSES = ["crack", "loss", "stain"]
STYLES = ["madhubani", "warli", "pattachitra", "kalamkari"]
SOURCE_FIELDS = ["file", "style", "source", "url", "license", "title", "credit"]


def append_sources(meta_path, rows):
    """Append provenance rows (file, style, source, url, license, title, credit) to data/sources.csv."""
    import csv
    meta_path = Path(meta_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    new = not meta_path.exists()
    with open(meta_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SOURCE_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in SOURCE_FIELDS})


# ----------------------------------------------------------------------------- io
def list_images(root):
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def imread_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite_rgb(path, rgb):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise IOError(f"cannot write image: {path}")


def imwrite_gray(path, gray):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), gray):
        raise IOError(f"cannot write image: {path}")


def resize_long_side(rgb, max_side):
    h, w = rgb.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return rgb
    f = max_side / s
    return cv2.resize(rgb, (int(round(w * f)), int(round(h * f))), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------- features
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


def infill_mask(prob, thr=0.5, dilate=5, min_area=20):
    """Union of all damage classes -> cleaned, dilated binary mask (uint8 0/255)."""
    m = (prob.max(axis=2) > thr).astype(np.uint8)
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


# ----------------------------------------------------------------------------- inpainting
def telea(rgb, mask255, radius=3):
    return cv2.inpaint(rgb, mask255, radius, cv2.INPAINT_TELEA)


class Lama:
    """Pretrained Big-LaMa through the `simple-lama-inpainting` package (weights auto-download)."""

    def __init__(self):
        from simple_lama_inpainting import SimpleLama
        self.m = SimpleLama()

    def __call__(self, rgb, mask255):
        from PIL import Image
        out = np.array(self.m(Image.fromarray(rgb), Image.fromarray(mask255).convert("L")))
        h, w = rgb.shape[:2]
        return out[:h, :w]


def try_lama():
    try:
        return Lama()
    except Exception as e:  # noqa: BLE001
        print(f"[warn] LaMa unavailable ({type(e).__name__}: {e}). Continuing with Telea only.")
        return None


def composite(orig, filled, mask255, feather=1.5):
    a = cv2.GaussianBlur((mask255 > 0).astype(np.float32), (0, 0), feather)[..., None]
    out = filled.astype(np.float32) * a + orig.astype(np.float32) * (1 - a)
    return np.clip(out, 0, 255).round().astype(np.uint8)


# ----------------------------------------------------------------------------- colour
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


def upscale(rgb, scale, device="cpu"):
    """Real-ESRGAN if installed (see README), otherwise bicubic with a warning."""
    try:
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        up = RealESRGANer(scale=4, model_path=url, model=net, tile=512, tile_pad=10, pre_pad=0,
                          half=str(device).startswith("cuda") and torch.cuda.is_available())
        out, _ = up.enhance(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), outscale=scale)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] Real-ESRGAN unavailable ({type(e).__name__}: {e}); using bicubic instead.")
        h, w = rgb.shape[:2]
        return cv2.resize(rgb, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


# ----------------------------------------------------------------------------- metrics
def dice_np(pred, gt, eps=1e-6):
    pred, gt = pred.astype(bool), gt.astype(bool)
    return float((2 * (pred & gt).sum() + eps) / (pred.sum() + gt.sum() + eps))
