"""Human-audit sheets for the professor's real test set (no ground truth exists, so humans are the only precision metric).

Part 1 (detections): N random detected-damage components -> numbered sheets: [crop | crop with detection outlined].
  Raters label each id: damage / design / unsure.   -> detector precision
Part 2 (A/B): M of those detections are inpainted by Telea, LaMa(fine-tuned) and Stable Diffusion; the three results are shown
  in a random order as A/B/C on the same crop. Raters pick the best (or 'same').   -> blind preference

    python scripts/eval/audit_make.py --images ../Dataset --ckpt runs/unet_v3/best.pt --out audit
Writes audit/sheets_det_*.jpg, audit/sheets_ab_*.jpg, audit/detections.csv, audit/ab_key.csv (keep the key away from raters),
and blank audit/labels_template.csv / audit/ab_template.csv.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from folk_restore import (composite, imread_rgb, imwrite_rgb, infill_mask, list_images, load_model, resize_long_side,
                          segment_image, telea)
from folk_restore.inpaint import DiffusionInpaint, Lama

CROP = 192


def crop_at(img, cy, cx, size):
    h, w = img.shape[:2]
    y0 = int(np.clip(cy - size // 2, 0, max(0, h - size)))
    x0 = int(np.clip(cx - size // 2, 0, max(0, w - size)))
    return y0, x0, img[y0:y0 + size, x0:x0 + size]


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (46, 22), (255, 255, 255), -1)
    cv2.putText(out, text, (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return out


def sheet(tiles, cols, path):
    rows = [np.hstack(tiles[i:i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i:i + cols]))) for i in range(0, len(tiles), cols)]
    imwrite_rgb(path, np.vstack(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--ckpt", default="runs/unet_v3/best.pt")
    ap.add_argument("--lama_ft", default="runs/lama_ft/best.pt")
    ap.add_argument("--out", default="audit")
    ap.add_argument("--n_det", type=int, default=30)
    ap.add_argument("--n_ab", type=int, default=15)
    ap.add_argument("--min_area", type=int, default=60)
    ap.add_argument("--max_side", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(a.seed)
    seg, mode = load_model(a.ckpt, device)

    comps, store = [], {}
    for p in list_images(a.images):
        img = resize_long_side(imread_rgb(p), a.max_side)
        mask = infill_mask(segment_image(seg, img, mode, device), 0.5, 7)
        raw = (mask > 0).astype(np.uint8)
        n, lab, stats, cen = cv2.connectedComponentsWithStats(raw, connectivity=8)
        store[p.stem] = (img, mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= a.min_area:
                comps.append((p.stem, i, int(stats[i, cv2.CC_STAT_AREA]), cen[i], lab))
    print(f"{len(comps)} detected components over {len(store)} images")
    pick = [comps[i] for i in rng.permutation(len(comps))[:a.n_det]]

    det_tiles, rows = [], []
    for k, (stem, ci, area, cen, lab) in enumerate(pick, 1):
        img, mask = store[stem]
        y0, x0, c = crop_at(img, int(cen[1]), int(cen[0]), CROP)
        m = (lab[y0:y0 + CROP, x0:x0 + CROP] == ci).astype(np.uint8)
        ov = c.copy()
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cnts, -1, (255, 0, 0), 1)
        det_tiles.append(label(np.hstack([c, ov]), str(k)))
        rows.append({"id": k, "image": stem, "cx": int(cen[0]), "cy": int(cen[1]), "area_px": area})
    for i in range(0, len(det_tiles), 10):
        sheet(det_tiles[i:i + 10], 2, out / f"sheets_det_{i // 10}.jpg")
    with open(out / "detections.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(out / "labels_template.csv", "w", newline="") as f:
        csv.writer(f).writerows([["id", "label (damage/design/unsure)"]] + [[r["id"], ""] for r in rows])

    # ---- part 2: blind A/B/C
    lama = Lama(a.lama_ft if Path(a.lama_ft).exists() else None, device)
    sd = DiffusionInpaint(device=device)
    ab_tiles, key = [], []
    for k, (stem, ci, area, cen, lab) in enumerate(pick[:a.n_ab], 1):
        img, mask = store[stem]
        y0, x0, c = crop_at(img, int(cen[1]), int(cen[0]), 512)
        m = mask[y0:y0 + 512, x0:x0 + 512]
        outs = {"telea": telea(c, m), "lama_ft": composite(c, lama(c, m), m), "diffusion": composite(c, sd(c, m), m)}
        order = list(rng.permutation(list(outs)))
        panel = [label(cv2.resize(c, (320, 320)), "in")] + [label(cv2.resize(outs[o], (320, 320)), "ABC"[j]) for j, o in enumerate(order)]
        ab_tiles.append(np.hstack(panel))
        key.append({"id": k, "A": order[0], "B": order[1], "C": order[2]})
        imwrite_rgb(out / f"ab_{k:02d}.png", np.hstack(panel))
    for i in range(0, len(ab_tiles), 5):
        sheet(ab_tiles[i:i + 5], 1, out / f"sheets_ab_{i // 5}.jpg")
    with open(out / "ab_key.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "A", "B", "C"]); w.writeheader(); w.writerows(key)
    with open(out / "ab_template.csv", "w", newline="") as f:
        csv.writer(f).writerows([["id", "best (A/B/C/same)"]] + [[r["id"], ""] for r in key])
    print("wrote", out)


if __name__ == "__main__":
    main()
