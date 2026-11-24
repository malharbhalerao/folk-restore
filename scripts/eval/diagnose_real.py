"""Diagnose why the segmenter finds little on REAL images: image sizes, damage % at several thresholds and several
working scales (the segmenter was trained on 512 px tiles; real photos may show damage at a very different scale).

    python scripts/eval/diagnose_real.py data/real_damaged --ckpt runs/unet_v3/best.pt --out results/diagnose
Writes <out>/diagnose.csv and one panel per image (rows = scale, cols = threshold) for the first --panels images.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from folk_restore import imwrite_rgb, imread_rgb, list_images, load_model, segment_image

SCALES = (0.5, 0.75, 1.0, 1.5, 2.0)
THRS = (0.2, 0.3, 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--ckpt", default="runs/unet_v3/best.pt")
    ap.add_argument("--out", default="results/diagnose")
    ap.add_argument("--max_side", type=int, default=1536)
    ap.add_argument("--panels", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mode = load_model(a.ckpt, device)
    paths = list_images(a.input)
    if a.limit:
        paths = paths[:a.limit]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, p in enumerate(paths):
        rgb0 = imread_rgb(p)
        H0, W0 = rgb0.shape[:2]
        tiles = []
        for s in SCALES:
            long0 = max(H0, W0)
            tgt = min(a.max_side * s, 3000)
            f = tgt / long0
            rgb = cv2.resize(rgb0, (max(32, int(W0 * f)), max(32, int(H0 * f))), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
            prob = segment_image(model, rgb, mode, device)
            pm = prob.max(axis=2)
            row = {"image": Path(p).name, "orig_wh": f"{W0}x{H0}", "scale": s, "work_wh": f"{rgb.shape[1]}x{rgb.shape[0]}"}
            for t in THRS:
                row[f"pct@{t}"] = round(100 * float((pm > t).mean()), 2)
            row["max_prob"] = round(float(pm.max()), 3)
            rows.append(row)
            if n < a.panels:
                sm = cv2.resize(rgb, (320, int(320 * rgb.shape[0] / rgb.shape[1])))
                cols = [sm]
                for t in THRS:
                    m = cv2.resize(((pm > t) * 255).astype(np.uint8), (sm.shape[1], sm.shape[0]), interpolation=cv2.INTER_NEAREST)
                    ov = sm.copy(); ov[m > 0] = (0.5 * ov[m > 0] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
                    cols.append(ov)
                tiles.append(np.hstack(cols))
        if tiles:
            hmax = max(t.shape[0] for t in tiles)
            tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
            imwrite_rgb(out / f"{Path(p).stem}_scales.png", np.vstack(tiles))
        print(Path(p).name, f"{W0}x{H0}", " | ".join(f"x{r['scale']}: {r['pct@0.5']}%" for r in rows[-len(SCALES):]))
    with open(out / "diagnose.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("wrote", out / "diagnose.csv", "(panels: rows = work scale 0.5..2x, cols = image, thr 0.2, 0.3, 0.5)")


if __name__ == "__main__":
    main()
