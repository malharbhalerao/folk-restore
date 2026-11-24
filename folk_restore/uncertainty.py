"""Milestone 4, part 2: spatial confidence map for the restored image.

Four signals, each scaled to [0, 1] where 1 = "speculative / do not trust":
  entropy   how unsure the segmenter was whether a pixel is damaged at all (p near 0.5)
  fill      how deep inside a repaired hole the pixel is. A hairline crack has context on both sides; the middle of a
            large lacuna is pure invention
  disagree  disagreement between several inpainting runs of the same hole (original, h-flip, 180-degree rotation).
            If the model answers differently when it sees the picture mirrored, it is guessing
  disc      the PatchGAN discriminator (trained during LaMa fine-tuning) scores how "real painting" a patch looks;
            patches scoring below what real patches score are suspicious
Outside the repaired region the pixels are the original paint, so only `entropy` (a missed-damage warning) applies.
Confidence = 1 - uncertainty. WEIGHTS below are the initial judgement-call weights; scripts/eval/uncertainty_sr.py fits better ones (fit_weights) on a validation split; scripts/eval/uncertainty_sr.py reports how well
each signal (and the fusion) predicts real error, so re-weight there if you disagree.
"""
import cv2
import numpy as np

WEIGHTS = {"entropy": 0.15, "fill": 0.20, "disagree": 0.30, "disc": 0.35}


def signal_entropy(prob):
    p = np.clip(1 - np.prod(1 - prob, axis=2), 1e-6, 1 - 1e-6)      # P(any damage)
    return (-(p * np.log2(p) + (1 - p) * np.log2(1 - p))).astype(np.float32)


def signal_fill(mask255, d0=10.0):
    m = (mask255 > 0).astype(np.uint8)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    return (1 - np.exp(-dist / d0)).astype(np.float32) * m


def _lab(rgb):
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def tta_inpaint(inpaint_fn, rgb, mask255):
    """Run an inpainter on the image, its horizontal mirror and its 180-degree rotation. Returns the list of 3 results,
    each mapped back to the original orientation. The first one (no transform) is the one used as the actual restoration."""
    outs = [inpaint_fn(rgb, mask255)]
    h = inpaint_fn(np.ascontiguousarray(rgb[:, ::-1]), np.ascontiguousarray(mask255[:, ::-1]))
    outs.append(np.ascontiguousarray(h[:, ::-1]))
    r = inpaint_fn(np.ascontiguousarray(rgb[::-1, ::-1]), np.ascontiguousarray(mask255[::-1, ::-1]))
    outs.append(np.ascontiguousarray(r[::-1, ::-1]))
    return outs


def signal_disagree(variants, mask255, scale=12.0):
    """Mean per-pixel distance (CIELAB units) of the variants from their own mean, scaled so `scale` units -> 1."""
    labs = np.stack([_lab(v) for v in variants])
    spread = np.sqrt(((labs - labs.mean(0, keepdims=True)) ** 2).sum(-1)).mean(0)
    spread = cv2.GaussianBlur(spread.astype(np.float32), (0, 0), 2.0)
    return np.clip(spread / scale, 0, 1) * (mask255 > 0)


def signal_disc(logits, calib):
    """logits: HxW discriminator map. calib: {'q50','q01'} = median / 1st percentile of logits on REAL (clean) patches.
    Real-median patch -> 0, real-1st-percentile or worse -> 1."""
    q50, q01 = calib["q50"], calib["q01"]
    return np.clip((q50 - logits) / max(q50 - q01, 1e-3), 0, 1).astype(np.float32)


def calibrate_disc(disc_logit_maps):
    """disc_logit_maps: list of HxW maps from CLEAN tiles -> calibration dict."""
    v = np.concatenate([m.ravel() for m in disc_logit_maps])
    return {"q50": float(np.percentile(v, 50)), "q01": float(np.percentile(v, 1)), "n_values": int(v.size)}


def fit_weights(pool, err, names=("entropy", "fill", "disagree", "disc")):
    """Non-negative least squares from per-pixel signals to the true restoration error (|RGB| / 255), on repaired pixels of a
    VALIDATION split. Returns {'weights': {...}, 'scale': 95th percentile of the fitted error}, used by fuse()."""
    from scipy.optimize import nnls
    A = np.stack([pool[k] for k in names], 1).astype(np.float64)
    w, _ = nnls(A, err.astype(np.float64) / 255.0)
    pred = A @ w
    return {"weights": {k: float(v) for k, v in zip(names, w)}, "scale": float(max(np.percentile(pred, 95), 1e-6))}


def load_fusion(path):
    """Fitted fusion weights (json from fit_weights) or None."""
    import json
    from pathlib import Path
    return json.load(open(path)) if path and Path(path).exists() else None


def fuse(signals, mask255, weights=None, smooth=1.5, scale=None):
    """signals: dict with any of entropy/fill/disagree/disc (HxW float32). Missing ones are skipped and the weights renormalised.
    Returns (confidence HxW float32 in [0,1], uncertainty HxW float32)."""
    w = dict(WEIGHTS if weights is None else weights)
    inside = (mask255 > 0).astype(np.float32)
    inside_soft = cv2.GaussianBlur(inside, (0, 0), 2.0)
    avail = [k for k in w if k in signals and signals[k] is not None]
    if scale is None:
        wsum = sum(w[k] for k in avail) or 1.0
        u_in = sum(w[k] * signals[k] for k in avail) / wsum
    else:                                           # fitted weights predict error directly; scale maps it to [0, 1]
        u_in = np.clip(sum(w[k] * signals[k] for k in avail) / scale, 0, 1)
    u_out = signals["entropy"] if signals.get("entropy") is not None else np.zeros_like(inside)
    u = inside_soft * u_in + (1 - inside_soft) * u_out
    if smooth:
        u = cv2.GaussianBlur(u.astype(np.float32), (0, 0), smooth)
    u = np.clip(u, 0, 1).astype(np.float32)
    return 1 - u, u


def heatmap(conf):
    """Confidence -> colour image (RGB uint8): green = trusted, red = speculative."""
    u8 = np.clip((1 - conf) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def overlay(rgb, conf, alpha=0.55):
    """Restored image with the repaired / low-confidence areas tinted red. Fully trusted pixels are left untouched."""
    u = (1 - conf)[..., None]
    red = np.array([255, 40, 40], np.float32)
    out = rgb.astype(np.float32) * (1 - alpha * u) + red * (alpha * u)
    return np.clip(out, 0, 255).astype(np.uint8)


def upsample_conf(conf, size_hw):
    h, w = size_hw
    return np.clip(cv2.resize(conf, (w, h), interpolation=cv2.INTER_CUBIC), 0, 1).astype(np.float32)
