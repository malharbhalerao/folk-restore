"""Step 1: raw images -> deduplicated, split-by-image, 512x512 tiles + manifest.csv

Expected input layout (either works):
    data/raw/<style>/*.jpg      e.g. data/raw/madhubani/..., data/raw/warli/...
    data/raw/*.jpg              (style recorded as 'all')
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import imread_rgb, imwrite_rgb, list_images


def dhash(rgb):
    g = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).flatten()


def load_prepared(path, min_side, max_side, tile):
    img = imread_rgb(path)
    h, w = img.shape[:2]
    if min(h, w) < min_side:
        return None
    s = max(h, w)
    if s > max_side:
        f = max_side / s
        img = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    if min(h, w) < tile:
        f = tile / min(h, w)
        img = cv2.resize(img, (int(np.ceil(w * f)), int(np.ceil(h * f))), interpolation=cv2.INTER_CUBIC)
    return img


def positions(n, tile, stride):
    pos = list(range(0, n - tile + 1, stride))
    if pos[-1] != n - tile:
        pos.append(n - tile)
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--stride", type=int, default=384)
    ap.add_argument("--min_side", type=int, default=600, help="skip images whose short side is smaller")
    ap.add_argument("--max_side", type=int, default=3000, help="downscale larger images (speed)")
    ap.add_argument("--max_tiles", type=int, default=40, help="max tiles kept per image")
    ap.add_argument("--min_std", type=float, default=12.0, help="drop flat/blank tiles")
    ap.add_argument("--hash_thr", type=int, default=6, help="Hamming distance for near-duplicates")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    raw, out = Path(a.raw), Path(a.out)
    files = list_images(raw)
    if not files:
        raise SystemExit(f"No images found under {raw}. Put the dataset there first.")
    print(f"found {len(files)} files")

    # ---- pass 1: filter + dedupe
    kept, hashes, skipped_small, dupes = [], [], 0, 0
    for p in files:
        try:
            img = load_prepared(p, a.min_side, a.max_side, a.tile)
        except IOError:
            continue
        if img is None:
            skipped_small += 1
            continue
        hsh = dhash(img)
        if any(np.count_nonzero(hsh != o) <= a.hash_thr for o in hashes):
            dupes += 1
            continue
        hashes.append(hsh)
        rel = p.relative_to(raw)
        kept.append({"path": p, "style": rel.parts[0] if len(rel.parts) > 1 else "all"})
    print(f"kept {len(kept)} images ({skipped_small} too small, {dupes} near-duplicates)")
    if len(kept) < 3:
        raise SystemExit("Need at least 3 usable images.")

    # ---- split BY IMAGE, stratified by style
    rng = random.Random(a.seed)
    by_style = defaultdict(list)
    for i, k in enumerate(kept):
        k["image_id"] = f"{i:05d}"
        by_style[k["style"]].append(k)
    for lst in by_style.values():
        rng.shuffle(lst)
        n = len(lst)
        n_test = max(1, round(n * a.test_frac)) if n >= 5 else (1 if n >= 3 else 0)
        n_val = max(1, round(n * a.val_frac)) if n >= 5 else 0
        for j, k in enumerate(lst):
            k["split"] = "test" if j < n_test else ("val" if j < n_test + n_val else "train")
    if not any(k["split"] == "val" for k in kept):
        next(k for k in kept if k["split"] == "train")["split"] = "val"

    # ---- pass 2: tile
    rows = []
    for k in kept:
        img = load_prepared(k["path"], a.min_side, a.max_side, a.tile)
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cand = []
        for y in positions(H, a.tile, a.stride):
            for x in positions(W, a.tile, a.stride):
                if gray[y:y + a.tile, x:x + a.tile].std() >= a.min_std:
                    cand.append((y, x))
        rng.shuffle(cand)
        for y, x in cand[:a.max_tiles]:
            tid = f"{k['image_id']}_{y}_{x}"
            rel = Path("tiles") / k["split"] / f"{tid}.jpg"
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / rel), cv2.cvtColor(img[y:y + a.tile, x:x + a.tile], cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append({"tile_id": tid, "path": str(rel), "image_id": k["image_id"],
                         "style": k["style"], "split": k["split"], "x": x, "y": y})

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\nsplit summary (images / tiles):")
    for s in ("train", "val", "test"):
        print(f"  {s:5s}: {sum(k['split'] == s for k in kept):4d} images, {sum(r['split'] == s for r in rows):6d} tiles")
    print("styles:", {s: len(v) for s, v in by_style.items()})
    print(f"manifest -> {out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
