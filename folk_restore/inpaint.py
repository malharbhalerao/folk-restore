"""Milestone 2: inpainting back-ends: OpenCV Telea, Big-LaMa (pretrained or our fine-tuned TorchScript), mask compositing."""
import cv2
import numpy as np
import torch


def telea(rgb, mask255, radius=3):
    return cv2.inpaint(rgb, mask255, radius, cv2.INPAINT_TELEA)


def _prep(rgb, mask255, device, mod=8):
    img = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    m = (mask255 > 0).astype(np.float32)[None]
    h, w = img.shape[1:]
    ph, pw = (-h) % mod, (-w) % mod
    if ph or pw:
        img = np.pad(img, ((0, 0), (0, ph), (0, pw)), mode="symmetric")
        m = np.pad(m, ((0, 0), (0, ph), (0, pw)), mode="symmetric")
    return torch.from_numpy(img)[None].to(device), torch.from_numpy((m > 0).astype(np.float32))[None].to(device), h, w


class Lama:
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


class DiffusionInpaint:
    """Stable Diffusion inpainting (diffusers), fp16, run on 512 px windows that contain mask pixels; the original is
    kept outside the mask. A different seed gives a different fill, so the spread over seeds is an uncertainty signal."""

    def __init__(self, model="stable-diffusion-v1-5/stable-diffusion-inpainting", device=None, steps=25, guidance=7.5,
                 prompt="a traditional Indian folk painting, intact, clean flat colours, fine detail",
                 negative="cracks, stains, damage, missing paint, blur"):
        from diffusers import StableDiffusionInpaintPipeline
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dt = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(model, torch_dtype=dt, safety_checker=None,
                                                                   requires_safety_checker=False).to(self.device)
        self.pipe.enable_attention_slicing()
        self.pipe.set_progress_bar_config(disable=True)
        self.kw = dict(num_inference_steps=steps, guidance_scale=guidance, prompt=prompt, negative_prompt=negative)

    def _window(self, rgb, mask255, seed):
        from PIL import Image
        h, w = rgb.shape[:2]
        ph, pw = (-h) % 8, (-w) % 8
        img = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect") if ph or pw else rgb
        msk = np.pad(mask255, ((0, ph), (0, pw)), mode="reflect") if ph or pw else mask255
        g = torch.Generator(device=self.device).manual_seed(seed)
        out = self.pipe(image=Image.fromarray(img), mask_image=Image.fromarray(msk), height=img.shape[0], width=img.shape[1],
                        generator=g, **self.kw).images[0]
        return np.array(out)[:h, :w]

    def __call__(self, rgb, mask255, seed=0, win=512):
        H, W = rgb.shape[:2]
        out = rgb.copy()
        for y in range(0, H, win):
            for x in range(0, W, win):
                m = mask255[y:y + win, x:x + win]
                if m.any():
                    out[y:y + win, x:x + win] = self._window(rgb[y:y + win, x:x + win], m, seed)
        return out


def try_lama(path=None):
    try:
        return Lama(path)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] LaMa unavailable ({type(e).__name__}: {e}). Continuing with Telea only.")
        return None


def composite(orig, filled, mask255, feather=1.5):
    a = cv2.GaussianBlur((mask255 > 0).astype(np.float32), (0, 0), feather)[..., None]
    out = filled.astype(np.float32) * a + orig.astype(np.float32) * (1 - a)
    return np.clip(out, 0, 255).round().astype(np.uint8)


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
