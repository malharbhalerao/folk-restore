"""Synthetic damage engine: clean tile in -> damaged tile + masks out.

Defects: stains/foxing, flaking losses, crack networks, and colour fading.
"""
import cv2
import numpy as np


def smooth_noise(h, w, rng, scales=(3, 6, 12, 24, 48)):
    out = np.zeros((h, w), np.float32)
    for i, s in enumerate(scales):
        n = rng.random((s, s)).astype(np.float32)
        out += cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC) / (1.6 ** i)
    out -= out.min()
    out /= out.max() + 1e-6
    return out


def paper_color(img):
    g = img.mean(axis=2)
    sel = img[g >= np.percentile(g, 80)]
    return np.median(sel, axis=0)


# ------------------------------------------------------------------ cracks
def _walk(mask, rng, x, y, ang, steps, width, depth):
    if steps < 3:
        return
    pts = [(x, y)]
    for _ in range(steps):
        ang += rng.normal(0, 0.3)
        x += 4 * np.cos(ang)
        y += 4 * np.sin(ang)
        pts.append((x, y))
        if depth < 3 and rng.random() < 0.05:
            _walk(mask, rng, x, y, ang + rng.choice([-1, 1]) * rng.uniform(0.4, 1.2),
                  steps // 2, max(1, width - 1), depth + 1)
    cv2.polylines(mask, [np.array(pts, np.int32).reshape(-1, 1, 2)], False, 255,
                  thickness=int(width), lineType=cv2.LINE_AA)


def apply_cracks(img, rng, n_roots):
    h, w = img.shape[:2]
    m = np.zeros((h, w), np.uint8)
    for _ in range(n_roots):
        _walk(m, rng, rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0, 2 * np.pi),
              int(rng.integers(40, 90)), int(rng.integers(1, 4)), 0)
    a = cv2.GaussianBlur(m, (0, 0), 0.7).astype(np.float32)[..., None] / 255.0 * rng.uniform(0.5, 0.9)
    dark = np.array([55, 40, 30], np.float32) + rng.normal(0, 8, 3)
    return img * (1 - a) + dark * a, (m > 60).astype(np.uint8)


# ------------------------------------------------------------------ flaking / losses
def apply_loss(img, rng, frac):
    h, w = img.shape[:2]
    n = smooth_noise(h, w, rng, (3, 5, 9, 17, 33)) + 0.15 * rng.random((h, w)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), 1.0)
    m = (n > np.quantile(n, 1 - frac)).astype(np.uint8)
    base = paper_color(img)
    fibre = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.2) * 6
    blotch = (smooth_noise(h, w, rng, (4, 8)) - 0.5) * 14
    fill = base[None, None, :] + (fibre + blotch)[..., None]
    a = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.8)[..., None]
    return img * (1 - a) + fill * a, m


# ------------------------------------------------------------------ stains / foxing
def apply_stains(img, rng):
    h, w = img.shape[:2]
    core = np.zeros((h, w), np.float32)  # spots + tide-line ring: this is what gets labelled
    for _ in range(int(rng.integers(3, 25))):
        cv2.circle(core, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(2, 14)),
                   float(rng.uniform(0.4, 1.0)), -1)
    inner = np.zeros((h, w), np.float32)  # faint tint inside a water ring (visible, not labelled)
    if rng.random() < 0.5:
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        axes = (int(rng.integers(40, 180)), int(rng.integers(30, 140)))
        ang = float(rng.uniform(0, 180))
        ring = np.zeros_like(core)
        cv2.ellipse(ring, c, axes, ang, 0, 360, 1.0, thickness=int(rng.integers(3, 7)))
        core = np.maximum(core, cv2.GaussianBlur(ring, (0, 0), 2.0) * 0.9)
        cv2.ellipse(inner, c, axes, ang, 0, 360, 0.25, -1)
    sigma, strength = float(rng.uniform(1.5, 3.0)), rng.uniform(0.35, 0.7)
    core = cv2.GaussianBlur(core, (0, 0), sigma) * strength
    s = np.maximum(core, cv2.GaussianBlur(inner, (0, 0), sigma) * strength)
    col = np.array([120, 85, 50], np.float32) + rng.normal(0, 10, 3)
    out = img * (1 - s[..., None] * (1 - col / 255.0))  # multiply-blend a brown tone
    return out, (core > 0.12).astype(np.uint8)


# ------------------------------------------------------------------ fading
def make_fade_strength(h, w, rng):
    lo, hi = rng.uniform(0.05, 0.3), rng.uniform(0.4, 0.8)
    return lo + (hi - lo) * smooth_noise(h, w, rng, (2, 3, 5))


def apply_fade(rgb, strength):
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    k = 1 - strength
    lab[..., 1] *= k
    lab[..., 2] = lab[..., 2] * k + strength * 8
    lab[..., 0] += strength * 12
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return (np.clip(out, 0, 1) * 255).round().astype(np.uint8)


# ------------------------------------------------------------------ main entry
def degrade(clean, rng, fade_prob=0.7, types=None):
    """clean: HxWx3 uint8 RGB. Returns dict:
        image        damaged + faded (what the pipeline sees)
        image_struct damaged, not faded (used to evaluate inpainting in isolation)
        faded_only   clean tile with only the fade applied (used to evaluate de-fading)
        masks        {'crack','loss','stain'} uint8 {0,1}
        faded        bool
    """
    h, w = clean.shape[:2]
    if types is None:
        k = int(rng.integers(1, 4))
        types = [str(t) for t in rng.choice(["crack", "loss", "stain"], size=k, replace=False)]
    img = clean.astype(np.float32)
    masks = {c: np.zeros((h, w), np.uint8) for c in ("crack", "loss", "stain")}
    if "stain" in types:
        img, masks["stain"] = apply_stains(img, rng)
    if "loss" in types:
        img, masks["loss"] = apply_loss(img, rng, float(rng.uniform(0.01, 0.15)))
    if "crack" in types:
        img, masks["crack"] = apply_cracks(img, rng, int(rng.integers(1, 5)))
    masks["stain"][masks["loss"] > 0] = 0
    struct = np.clip(img, 0, 255).round().astype(np.uint8)

    strength = make_fade_strength(h, w, rng)
    do_fade = bool(rng.random() < fade_prob)
    return dict(
        image=apply_fade(struct, strength) if do_fade else struct,
        image_struct=struct,
        faded_only=apply_fade(clean, strength) if do_fade else clean.copy(),
        masks=masks,
        faded=do_fade,
    )


# =================================================================== v2 (domain-gap fix)
# v1 only ever showed the segmenter *damaged vs. plain* tiles, so dark printed/painted design lines looked like cracks.
# v2 adds (a) distractor design strokes that are NOT labelled, (b) fully clean tiles as hard negatives,
# (c) photographic variation (glare, gamma, blur, JPEG, noise) applied to the network input only.
INK = np.array([[25, 20, 20], [30, 30, 70], [70, 35, 25], [110, 30, 30], [20, 50, 40], [55, 40, 30]], np.float32)


def _bezier(rng, h, w, n=60):
    p = rng.uniform([0, 0], [w, h], (4, 2))
    t = np.linspace(0, 1, n)[:, None]
    pts = ((1 - t) ** 3) * p[0] + 3 * ((1 - t) ** 2) * t * p[1] + 3 * (1 - t) * t ** 2 * p[2] + t ** 3 * p[3]
    return pts.astype(np.int32).reshape(-1, 1, 2)


def add_distractors(img, rng):
    """Draw design-like dark strokes (smooth vines, ring motifs, dot rows, leaves) onto img. Not damage; no mask."""
    h, w = img.shape[:2]
    layer = np.zeros((h, w), np.uint8)
    col = INK[rng.integers(len(INK))] + rng.normal(0, 6, 3)
    for _ in range(int(rng.integers(1, 5))):
        kind = rng.choice(["vine", "ring", "dots", "leaf", "wave"])
        t = int(rng.integers(1, 5))
        if kind == "vine":
            cv2.polylines(layer, [_bezier(rng, h, w)], False, 255, t, cv2.LINE_AA)
        elif kind == "ring":
            c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            r0 = int(rng.integers(8, 60))
            for k in range(int(rng.integers(1, 4))):
                cv2.circle(layer, c, r0 + 6 * k, 255, t, cv2.LINE_AA)
        elif kind == "dots":
            x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
            a = rng.uniform(0, 2 * np.pi)
            step = rng.uniform(8, 16)
            for k in range(int(rng.integers(5, 20))):
                cv2.circle(layer, (int(x0 + k * step * np.cos(a)), int(y0 + k * step * np.sin(a))),
                           int(rng.integers(1, 4)), 255, -1, cv2.LINE_AA)
        elif kind == "leaf":
            for _ in range(int(rng.integers(2, 8))):
                cv2.ellipse(layer, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                            (int(rng.integers(6, 22)), int(rng.integers(2, 8))), float(rng.uniform(0, 180)),
                            0, 360, 255, -1 if rng.random() < 0.5 else 1, cv2.LINE_AA)
        else:
            xs = np.arange(0, w, 3)
            y0, amp, fr = rng.uniform(0, h), rng.uniform(4, 14), rng.uniform(0.03, 0.12)
            pts = np.stack([xs, y0 + amp * np.sin(xs * fr)], 1).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(layer, [pts], False, 255, t, cv2.LINE_AA)
    a = cv2.GaussianBlur(layer, (0, 0), 0.6).astype(np.float32)[..., None] / 255.0 * rng.uniform(0.7, 1.0)
    return img * (1 - a) + col * a


def photometric(img, rng):
    """Photographic variation on the network input: gamma/contrast, glare, blur, noise, JPEG."""
    h, w = img.shape[:2]
    x = img.astype(np.float32) / 255.0
    x = np.clip(x, 0, 1) ** rng.uniform(0.75, 1.3)
    x = (x - 0.5) * rng.uniform(0.8, 1.2) + 0.5 + rng.uniform(-0.06, 0.06)
    if rng.random() < 0.4:   # flash glare / uneven light
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        yy, xx = np.mgrid[0:h, 0:w]
        g = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (rng.uniform(0.3, 0.9) * max(h, w)) ** 2)))
        x = x * (1 - 0.35 * rng.uniform(0, 1) * (1 - g)[..., None]) + 0.15 * g[..., None] * rng.uniform(0, 1)
    if rng.random() < 0.4:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.4, 1.4))
    x = x + rng.normal(0, rng.uniform(0, 0.02), x.shape)
    out = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    if rng.random() < 0.5:
        ok, enc = cv2.imencode(".jpg", out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 90))])
        out = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1].copy()
    return out


def degrade_v2(clean, rng, fade_prob=0.6, p_clean=0.2, p_distract=0.6, photo=True):
    """Like degrade(), plus distractor strokes and clean negatives. Also returns 'clean_aug' (clean + distractors)."""
    h, w = clean.shape[:2]
    base = clean
    if rng.random() < p_distract:
        base = np.clip(add_distractors(clean.astype(np.float32), rng), 0, 255).round().astype(np.uint8)
    if rng.random() < p_clean:
        d = dict(image=base.copy(), image_struct=base.copy(), faded_only=base.copy(), faded=False,
                 masks={c: np.zeros((h, w), np.uint8) for c in ("crack", "loss", "stain")})
        if rng.random() < fade_prob:
            s = make_fade_strength(h, w, rng)
            d["image"], d["faded_only"], d["faded"] = apply_fade(base, s), apply_fade(base, s), True
    else:
        d = degrade(base, rng, fade_prob=fade_prob)
    if photo:
        d["image"] = photometric(d["image"], rng)
    d["clean_aug"] = base
    return d
