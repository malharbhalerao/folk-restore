"""run the full pipeline on whole images (real damaged paintings).

    python scripts/restore_real.py --input_dir data/real_damaged
Optional: --defade_model runs/defade/best.pt (scripts/train/defade.py), --sr 2 (Real-ESRGAN, see README).

Outputs per image (in --out):
  <name>_input.png  <name>_mask.png  <name>_unc.png  <name>_telea.png  <name>_lama.png
  [<name>_defade.png]  [<name>_sr.png]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from folk_restore import (composite, defade_image, imread_rgb, load_defade, imwrite_gray, imwrite_rgb, infill_mask, list_images, load_model,
                    resize_long_side, reinhard, segment_image, telea, try_lama, upscale)


def entropy_map(prob):
    p = np.clip(1 - np.prod(1 - prob, axis=2), 1e-6, 1 - 1e-6)  # P(any damage)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))          # 0..1 bits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="data/real_damaged")
    ap.add_argument("--pattern", default=None, help="optional glob filter, e.g. '*_damaged.png'")
    ap.add_argument("--ckpt", default="runs/unet_rich/best.pt")
    ap.add_argument("--out", default="results/real")
    ap.add_argument("--max_side", type=int, default=1536, help="working resolution (long side)")
    ap.add_argument("--dilate", type=int, default=7)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--defade_model", default=None, help="learned de-fader from step 8, e.g. runs/defade/best.pt (recommended)")
    ap.add_argument("--defade", default=None, help="OLD global Reinhard baseline (ref_stats.npz); it made colours WORSE in our test, avoid")
    ap.add_argument("--defade_strength", type=float, default=0.7, help="0 = no colour change, 1 = full correction")
    ap.add_argument("--sr", type=int, default=0, help="super-resolution factor (2 or 4); 0 = off")
    ap.add_argument("--no_lama", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    files = list_images(a.input_dir)
    if a.pattern:
        files = [f for f in files if f.match(a.pattern)]
    if a.limit:
        files = files[:a.limit]
    if not files:
        raise SystemExit(f"No images in {a.input_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mode = load_model(a.ckpt, device)
    lama = None if a.no_lama else try_lama()
    ref = np.load(a.defade) if a.defade else None
    dmodel = load_defade(a.defade_model, device) if a.defade_model else None
    out = Path(a.out)

    for p in tqdm(files, desc="restoring"):
        name = p.stem
        img = resize_long_side(imread_rgb(p), a.max_side)
        prob = segment_image(model, img, mode, device)
        mask = infill_mask(prob, a.thr, a.dilate)
        unc = cv2.applyColorMap((entropy_map(prob) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

        imwrite_rgb(out / f"{name}_input.png", img)
        imwrite_gray(out / f"{name}_mask.png", mask)
        imwrite_rgb(out / f"{name}_unc.png", cv2.cvtColor(unc, cv2.COLOR_BGR2RGB))
        t = telea(img, mask)
        imwrite_rgb(out / f"{name}_telea.png", t)
        final = t
        if lama is not None:
            final = composite(img, lama(img, mask), mask)
            imwrite_rgb(out / f"{name}_lama.png", final)
        if dmodel is not None:
            final = defade_image(dmodel, final, device, a.defade_strength)
            imwrite_rgb(out / f"{name}_defade.png", final)
        elif ref is not None:
            final = reinhard(final, ref["mean"], ref["std"], a.defade_strength)
            imwrite_rgb(out / f"{name}_defade.png", final)
        if a.sr:
            imwrite_rgb(out / f"{name}_sr.png", upscale(final, a.sr, device))
        print(f"  {name}: damaged area = {100 * (mask > 0).mean():.1f}%")
    print(f"\nsaved to {out}/")


if __name__ == "__main__":
    main()
