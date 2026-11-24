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



# ------------------------------------------------------------------ decoys (hard negatives)
def apply_decoys(img, rng):
    """Draw UNLABELLED dark/light design lines on a clean tile: smooth vines, outlines, dotted borders.

    These look like cracks to a naive detector but are part of the painting, so they are NOT in any damage mask.
    Training on them teaches the segmenter that long, smooth, uniform-width strokes are design, while
    jagged, branching, tapering lines are cracks. Returns the modified (still "clean") image.
    """
    h, w = img.shape[:2]
    out = img.astype(np.float32).copy()
    layer = np.zeros((h, w), np.uint8)
    for _ in range(int(rng.integers(1, 4))):
        kind = rng.choice(["vine", "outline", "dots"])
        if kind == "dots":                                   # row of dots/dashes, typical of Warli / Madhubani borders
            x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
            ang, step, r = rng.uniform(0, 2 * np.pi), rng.uniform(8, 18), int(rng.integers(2, 5))
            for k in range(int(rng.integers(8, 25))):
                cv2.circle(layer, (int(x0 + k * step * np.cos(ang)), int(y0 + k * step * np.sin(ang))), r, 255, -1, cv2.LINE_AA)
        else:
            n = 4 if kind == "vine" else 6
            ctrl = np.stack([rng.uniform(0, w, n), rng.uniform(0, h, n)], 1)
            t = np.linspace(0, 1, 160)
            pts = np.zeros((160, 2))
            for i in range(n - 1):                           # Catmull-Rom-ish smooth path through the control points
                sel = (t >= i / (n - 1)) & (t <= (i + 1) / (n - 1))
                u = ((t[sel] - i / (n - 1)) * (n - 1))[:, None]
                pts[sel] = ctrl[i] * (1 - u) ** 2 + 2 * ((ctrl[i] + ctrl[i + 1]) / 2 + rng.normal(0, 25, 2)) * u * (1 - u) + ctrl[i + 1] * u ** 2
            cv2.polylines(layer, [pts.astype(np.int32).reshape(-1, 1, 2)], False, 255,
                          thickness=int(rng.integers(3, 9)), lineType=cv2.LINE_AA)
    a = cv2.GaussianBlur(layer, (0, 0), 0.8).astype(np.float32)[..., None] / 255.0 * rng.uniform(0.7, 1.0)
    colour = np.array([[50, 35, 30], [90, 40, 40], [20, 20, 20], [235, 225, 205]][int(rng.integers(0, 4))], np.float32)
    return np.clip(out * (1 - a) + colour * a, 0, 255).round().astype(np.uint8)



# ------------------------------------------------------------------ WALL-MURAL damage (v3)
PLASTER_PALETTE = np.array([[205, 195, 172], [226, 218, 200], [178, 150, 118], [150, 98, 76],
                            [128, 118, 105], [190, 172, 140]], np.float32)


def apply_plaster_loss(img, rng, frac):
    """Chalky plaster/paint flaking: large irregular holes that expose lime plaster, ochre or brick underlayer,
    with grainy texture and a thin shadow rim. Unlike apply_loss the fill colour does NOT come from the painting."""
    h, w = img.shape[:2]
    n = smooth_noise(h, w, rng, (2, 4, 7, 13, 26, 52)) + 0.25 * rng.random((h, w)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), 1.5)
    m = (n > np.quantile(n, 1 - frac)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    base = PLASTER_PALETTE[int(rng.integers(0, len(PLASTER_PALETTE)))] + rng.normal(0, 6, 3)
    grain = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.0) * rng.uniform(6, 16)
    patch = (smooth_noise(h, w, rng, (3, 6, 12)) - 0.5) * rng.uniform(20, 45)
    fill = base[None, None, :] + (grain + patch)[..., None]
    a = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.7)[..., None]
    out = img * (1 - a) + fill * a
    # shadow rim just outside the hole, offset down-right like raking light
    sh = cv2.dilate(m, np.ones((5, 5), np.uint8)) - m
    sh = np.roll(np.roll(sh, 2, 0), 2, 1).astype(np.float32)
    sh = cv2.GaussianBlur(sh, (0, 0), 1.2)[..., None] * 0.35
    return np.clip(out * (1 - sh), 0, 255), m


def apply_salt(img, rng):
    """Efflorescence / whitewash blooms: semi-opaque whitish blotches (labelled as 'stain')."""
    h, w = img.shape[:2]
    n = smooth_noise(h, w, rng, (3, 6, 12, 24)) + 0.2 * rng.random((h, w)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), 1.5)
    cov = rng.uniform(0.03, 0.2)
    a = np.clip((n - np.quantile(n, 1 - cov)) * 6, 0, 1) * rng.uniform(0.45, 0.85)
    colour = np.array([238, 232, 220], np.float32) + rng.normal(0, 5, 3)
    out = img * (1 - a[..., None]) + colour * a[..., None]
    return out, (a > 0.2).astype(np.uint8)


def apply_streaks(img, rng):
    """Rain-water runs: long, mostly vertical dark-brown streaks that fade towards their ends (labelled 'stain')."""
    h, w = img.shape[:2]
    layer = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(2, 9))):
        x, y0 = int(rng.integers(0, w)), int(rng.integers(0, h // 2))
        length = int(rng.integers(h // 4, h))
        wd = int(rng.integers(2, 12))
        sx = rng.normal(0, 0.15)
        pts = np.array([[x + sx * k + 3 * np.sin(k / 23.0), y0 + k] for k in range(0, length, 4)], np.int32)
        cv2.polylines(layer, [pts.reshape(-1, 1, 2)], False, float(rng.uniform(0.5, 1.0)), wd, cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (0, 0), 2.5) * rng.uniform(0.4, 0.75)
    col = np.array([95, 75, 55], np.float32) + rng.normal(0, 8, 3)
    out = img * (1 - layer[..., None] * (1 - col / 255.0))
    return out, (layer > 0.1).astype(np.uint8)


def photo_aug(rgb, rng):
    """Phone-photo look: uneven illumination, soft focus, sensor noise, JPEG blocking, mild colour cast."""
    h, w = rgb.shape[:2]
    x = rgb.astype(np.float32)
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    grad = ((gx / w - .5) * np.cos(ang) + (gy / h - .5) * np.sin(ang)) * rng.uniform(0, 0.35)
    x = x * (1 + grad)[..., None] * rng.uniform(0.8, 1.1) * (1 + rng.normal(0, 0.03, 3))
    if rng.random() < 0.6:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.4, 1.6))
    x = x + rng.normal(0, rng.uniform(0, 5), x.shape)
    x = np.clip(x, 0, 255).astype(np.uint8)
    if rng.random() < 0.6:
        ok, buf = cv2.imencode(".jpg", x[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 90))])
        x = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]
    return np.ascontiguousarray(x)

# ------------------------------------------------------------------ main entry
def _degrade_once(clean, rng, fade_prob=0.7, types=None, decoy_prob=0.0, profile='paper'):
    """clean: HxWx3 uint8 RGB. Returns dict:
        clean        the clean tile actually used (with decoy design lines if decoy_prob fired)
        image        damaged + faded (what the pipeline sees)
        image_struct damaged, not faded (used to evaluate inpainting in isolation)
        faded_only   clean tile with only the fade applied (used to evaluate de-fading)
        masks        {'crack','loss','stain'} uint8 {0,1}
        faded        bool
    """
    h, w = clean.shape[:2]
    if decoy_prob > 0 and rng.random() < decoy_prob:
        clean = apply_decoys(clean, rng)
    if types is None:
        k = int(rng.integers(1, 4))
        types = [str(t) for t in rng.choice(["crack", "loss", "stain"], size=k, replace=False)]
    img = clean.astype(np.float32)
    masks = {c: np.zeros((h, w), np.uint8) for c in ("crack", "loss", "stain")}
    wall = profile == "wall"
    if "stain" in types:
        img, masks["stain"] = apply_stains(img, rng)
        if wall:                                   # wall stains: also salt blooms and rain streaks
            if rng.random() < 0.6:
                img, m2 = apply_salt(img, rng); masks["stain"] |= m2
            if rng.random() < 0.5:
                img, m2 = apply_streaks(img, rng); masks["stain"] |= m2
    if "loss" in types:
        if wall:
            img, masks["loss"] = apply_plaster_loss(img, rng, float(rng.uniform(0.02, 0.30)))
        else:
            img, masks["loss"] = apply_loss(img, rng, float(rng.uniform(0.01, 0.15)))
    if "crack" in types:
        img, masks["crack"] = apply_cracks(img, rng, int(rng.integers(1, 5)))
    masks["stain"][masks["loss"] > 0] = 0
    struct = np.clip(img, 0, 255).round().astype(np.uint8)

    strength = make_fade_strength(h, w, rng)
    do_fade = bool(rng.random() < fade_prob)
    return dict(
        clean=clean,
        image=apply_fade(struct, strength) if do_fade else struct,
        image_struct=struct,
        faded_only=apply_fade(clean, strength) if do_fade else clean.copy(),
        masks=masks,
        faded=do_fade,
    )


def degrade(clean, rng, fade_prob=0.7, types=None, decoy_prob=0.0, profile="paper", scale_jitter=0.0, photo=0.0):
    """Public entry. profile: 'paper' (original), 'wall' (plaster flaking, salt, streaks) or 'mixed' (50/50 per tile).
    scale_jitter in [0,1]: probability that the damage is rendered at a random scale (0.5x-2x) so crack width and
    hole size vary relative to the painting (real photos come at unknown scales).
    photo: probability of phone-photo degradations (blur, noise, JPEG, uneven light) applied to the input image."""
    prof = profile if profile != "mixed" else ("wall" if rng.random() < 0.5 else "paper")
    h, w = clean.shape[:2]
    if scale_jitter > 0 and rng.random() < scale_jitter:
        s = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
        hh, ww = max(64, int(round(h / s))), max(64, int(round(w / s)))
        small = cv2.resize(clean, (ww, hh), interpolation=cv2.INTER_AREA if s > 1 else cv2.INTER_CUBIC)
        d = _degrade_once(small, rng, fade_prob, types, decoy_prob, prof)
        rs = lambda a, nn=False: cv2.resize(a, (w, h), interpolation=cv2.INTER_NEAREST if nn else cv2.INTER_CUBIC)
        d = dict(clean=clean if not decoy_prob else rs(d["clean"]), image=rs(d["image"]),
                 image_struct=rs(d["image_struct"]), faded_only=rs(d["faded_only"]),
                 masks={k: rs(v, True) for k, v in d["masks"].items()}, faded=d["faded"])
    else:
        d = _degrade_once(clean, rng, fade_prob, types, decoy_prob, prof)
    if photo > 0 and rng.random() < photo:
        d["image"] = photo_aug(d["image"], rng)
    d["profile"] = prof
    return d
