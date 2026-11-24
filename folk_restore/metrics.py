"""Evaluation metrics shared by the scripts."""
import numpy as np


def psnr_region(a, b, region):
    r = region > 0
    if r.sum() == 0:
        return None
    mse = np.mean((a[r].astype(np.float64) - b[r].astype(np.float64)) ** 2)
    return 99.0 if mse < 1e-9 else float(10 * np.log10(255.0 ** 2 / mse))


def psnr_full(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse < 1e-9 else float(10 * np.log10(255.0 ** 2 / mse))


def de00(a, b):
    """Mean CIEDE2000 between two RGB uint8 images (lower = better)."""
    from skimage.color import deltaE_ciede2000, rgb2lab
    return float(deltaE_ciede2000(rgb2lab(a), rgb2lab(b)).mean())


def make_lpips(device):
    """Returns f(x, y) -> LPIPS(AlexNet) between two RGB uint8 images, or a function returning None if unavailable."""
    try:
        import lpips
        import torch
        net = lpips.LPIPS(net="alex", verbose=False).to(device)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] LPIPS unavailable ({e}); skipping LPIPS")
        return lambda x, y: None

    def f(x, y):
        t = lambda im: torch.from_numpy(im).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(device)  # noqa: E731
        with torch.no_grad():
            return float(net(t(x), t(y)).item())
    return f
