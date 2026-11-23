"""Image and provenance I/O."""
from pathlib import Path

import cv2

from .config import IMG_EXTS, SOURCE_FIELDS


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

