"""Step 4: Milestone 1 + 2 numbers on the fixed synthetic TEST set.

Segmentation: Dice per class + union.
Inpainting  : Telea vs pretrained LaMa, with ORACLE masks (ground truth) and PREDICTED masks (our U-Net),
              scored on the damaged region (PSNR) and whole tile (SSIM, LPIPS), against the clean tile.
Inpainting is run on the *_struct.png image (damage without fading) so fading doesn't pollute the score.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity
from tqdm import tqdm

from common import (CLASSES, composite, dice_np, dilate_mask, imread_rgb, imwrite_gray, imwrite_rgb,
                    infill_mask, load_model, segment_image, telea, try_lama)


def psnr_region(a, b, region):
    r = region > 0
    if r.sum() == 0:
        return None
    mse = np.mean((a[r].astype(np.float64) - b[r].astype(np.float64)) ** 2)
    return 99.0 if mse < 1e-9 else float(10 * np.log10(255.0 ** 2 / mse))


def psnr_full(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse < 1e-9 else float(10 * np.log10(255.0 ** 2 / mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ckpt", default="runs/unet_rich/best.pt")
    ap.add_argument("--n", type=int, default=100, help="number of test tiles to evaluate")
    ap.add_argument("--dilate", type=int, default=5)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--save_n", type=int, default=12, help="save example images for the first N tiles")
    ap.add_argument("--out", default="results")
    ap.add_argument("--test_dir", default=None, help="synthetic test folder (default <data>/synthetic/test; use <data>/synthetic_wall/test)")
    ap.add_argument("--no_lpips", action="store_true")
    ap.add_argument("--no_lama", action="store_true")
    a = ap.parse_args()

    data, out = Path(a.data), Path(a.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mode = load_model(a.ckpt, device)
    lama = None if a.no_lama else try_lama()
    lp = None
    if not a.no_lpips:
        try:
            import lpips
            lp = lpips.LPIPS(net="alex", verbose=False).to(device)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] LPIPS unavailable ({e}); skipping LPIPS")

    def lpips_val(x, y):
        if lp is None:
            return None
        t = lambda im: torch.from_numpy(im).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(device)  # noqa: E731
        with torch.no_grad():
            return float(lp(t(x), t(y)).item())

    folder = Path(a.test_dir) if a.test_dir else data / "synthetic" / "test"
    idx = {r["tile_id"]: r for r in csv.DictReader(open(folder / "index.csv"))}
    ids = list(idx)[:a.n]
    if not ids:
        raise SystemExit("No synthetic test tiles. Run 02_make_synthetic.py first.")

    rows, seg_inter, seg_p, seg_g = [], np.zeros(3), np.zeros(3), np.zeros(3)
    union_dice = []
    for k, tid in enumerate(tqdm(ids, desc="evaluating")):
        clean = imread_rgb(data / idx[tid]["path"])
        damaged = imread_rgb(folder / f"{tid}_damaged.png")
        struct = imread_rgb(folder / f"{tid}_struct.png")
        gt = imread_rgb(folder / f"{tid}_mask.png") > 127          # HxWx3 bool
        gt_union = gt.any(axis=2).astype(np.uint8)

        # --- Milestone 1: segmentation on the damaged+faded tile
        prob = segment_image(model, damaged, mode, device)
        pred = prob > a.thr
        for c in range(3):
            seg_inter[c] += (pred[..., c] & gt[..., c]).sum()
            seg_p[c] += pred[..., c].sum()
            seg_g[c] += gt[..., c].sum()
        if gt_union.sum() > 0:
            union_dice.append(dice_np(pred.any(axis=2), gt_union))

        # --- Milestone 2: inpainting
        mask_oracle = dilate_mask(gt_union, a.dilate)
        mask_pred = infill_mask(prob, a.thr, a.dilate)
        region = mask_oracle  # score everything on the same region: the dilated ground-truth damage
        results = {"damaged (no restoration)": struct,
                   "Telea, oracle mask": telea(struct, mask_oracle),
                   "Telea, predicted mask": telea(struct, mask_pred)}
        if lama is not None:
            results["LaMa, oracle mask"] = composite(struct, lama(struct, mask_oracle), mask_oracle)
            results["LaMa, predicted mask"] = composite(struct, lama(struct, mask_pred), mask_pred)
        for name, img in results.items():
            rows.append({"tile_id": tid, "method": name,
                         "psnr_region": psnr_region(img, clean, region),
                         "psnr_full": psnr_full(img, clean),
                         "ssim_full": float(structural_similarity(img, clean, channel_axis=2)),
                         "lpips_full": lpips_val(img, clean)})
        if k < a.save_n:
            s = out / "samples"
            imwrite_rgb(s / f"{tid}_clean.png", clean)
            imwrite_rgb(s / f"{tid}_damaged.png", struct)
            imwrite_gray(s / f"{tid}_gtmask.png", mask_oracle)
            imwrite_gray(s / f"{tid}_mask.png", mask_pred)
            imwrite_rgb(s / f"{tid}_telea.png", results["Telea, predicted mask"])
            if lama is not None:
                imwrite_rgb(s / f"{tid}_lama.png", results["LaMa, predicted mask"])

    # ---- save + summarise
    eps = 1e-6
    seg = {f"dice_{c}": float((2 * seg_inter[i] + eps) / (seg_p[i] + seg_g[i] + eps)) for i, c in enumerate(CLASSES)}
    seg["dice_union_mean_per_tile"] = float(np.mean(union_dice)) if union_dice else None
    seg["n_tiles"] = len(ids)
    (out / "seg_metrics.json").write_text(json.dumps(seg, indent=2))
    with open(out / "inpaint_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    methods = list(dict.fromkeys(r["method"] for r in rows))
    lines = ["| Method | PSNR (damaged region) | PSNR (tile) | SSIM (tile) | LPIPS (tile, lower=better) |",
             "| --- | --- | --- | --- | --- |"]
    for m in methods:
        rr = [r for r in rows if r["method"] == m]
        mean = lambda key: np.mean([r[key] for r in rr if r[key] is not None]) if any(r[key] is not None for r in rr) else float("nan")  # noqa: E731
        lines.append(f"| {m} | {mean('psnr_region'):.2f} | {mean('psnr_full'):.2f} | {mean('ssim_full'):.4f} | {mean('lpips_full'):.4f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")

    print("\n=== Segmentation (synthetic test) ===")
    print(json.dumps(seg, indent=2))
    print("\n=== Inpainting (synthetic test) ===")
    print("\n".join(lines))
    print(f"\nsaved to {out}/ (seg_metrics.json, inpaint_metrics.csv, summary.md, samples/)")


if __name__ == "__main__":
    main()
