"""Milestone 2: inpainting back-ends (OpenCV Telea, pretrained Big-LaMa) and mask compositing."""
import cv2
import numpy as np


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

