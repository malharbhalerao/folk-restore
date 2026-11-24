"""Compose the report's qualitative figure.

Row 1: a synthetic test tile  (clean | damaged | Telea | LaMa fine-tuned | diffusion)  from results/<samples>/
Rows 2-3: real Mandawa photographs (input | detected damage | restored | confidence)   from restore.py outputs.

    python scripts/make_report_figures.py --synthetic_dir results/samples_ft --restored results/restored_mandawa \
        --images ../Dataset --names mandawa-19mar26_2 mandawa-19mar20_12
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from folk_restore import imread_rgb, imwrite_rgb, resize_long_side

H = 240
CROP = 420


def fit(img, h=H):
    return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h), interpolation=cv2.INTER_AREA)


def caption(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, out.shape[0] - 20), (out.shape[1], out.shape[0]), (255, 255, 255), -1)
    cv2.putText(out, text, (4, out.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def pad_to(row, w):
    return cv2.copyMakeBorder(row, 0, 0, 0, max(0, w - row.shape[1]), cv2.BORDER_CONSTANT, value=(255, 255, 255))[:, :w]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic_dir", default="results/samples_ft")
    ap.add_argument("--tile", default=None, help="synthetic tile id (default: first with a diffusion image)")
    ap.add_argument("--restored", default="results/restored_mandawa")
    ap.add_argument("--images", default="../Dataset")
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", default="report/figures/qualitative.jpg")
    a = ap.parse_args()
    rows = []
    sd = Path(a.synthetic_dir)
    tid = a.tile or sorted(p.name[:-len("_diffusion.png")] for p in sd.glob("*_diffusion.png"))[0]
    rows.append(np.hstack([caption(fit(imread_rgb(sd / f"{tid}_{k}.png")), c) for k, c in
                           [("clean", "clean"), ("damaged", "damaged"), ("telea", "Telea"), ("lama", "LaMa fine-tuned"), ("diffusion", "diffusion")]]))
    for n in a.names:
        r = Path(a.restored)
        img = resize_long_side(imread_rgb(Path(a.images) / f"{n}.jpg"), 1536)
        mask = imread_rgb(r / f"{n}_mask.png")[..., 0]
        # square crop (CROP px at working resolution) centred on the largest detected component
        cnt, lab, st, cen = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])) if cnt > 1 else 0
        cx, cy = (int(cen[k][0]), int(cen[k][1])) if cnt > 1 else (img.shape[1] // 2, img.shape[0] // 2)
        y0 = int(np.clip(cy - CROP // 2, 0, max(0, img.shape[0] - CROP)))
        x0 = int(np.clip(cx - CROP // 2, 0, max(0, img.shape[1] - CROP)))
        box = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
        big = (slice(2 * y0, 2 * (y0 + CROP)), slice(2 * x0, 2 * (x0 + CROP)))     # restored/confidence are x2
        ov = img.copy()
        ov[mask > 0] = (0.4 * ov[mask > 0] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
        rows.append(np.hstack([caption(fit(img[box]), "input (crop)"), caption(fit(ov[box]), "detected damage"),
                               caption(fit(imread_rgb(r / f"{n}_restored.png")[big]), "restored (x2)"),
                               caption(fit(imread_rgb(r / f"{n}_confidence.png")[big]), "confidence (red = speculative)")]))
    w = max(r.shape[1] for r in rows)
    imwrite_rgb(a.out, np.vstack([pad_to(r, w) for r in rows]))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
