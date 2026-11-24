"""fine-tuned LaMa vs zero-shot LaMa vs Telea on the fixed synthetic TEST set.

Same protocol as step 4 (inpainting run on the un-faded damaged tile; scored on the dilated ground-truth damage region and
on the whole tile), with both ORACLE masks and our PREDICTED masks. Adds the fine-tuned model as extra rows.

    python scripts/eval/inpaint.py --lama_ft runs/lama_ft/best.pt
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity
from tqdm import tqdm

from folk_restore import composite, dilate_mask, imread_rgb, imwrite_gray, imwrite_rgb, infill_mask, load_model, segment_image, telea
from folk_restore.inpaint import Lama
from folk_restore.metrics import make_lpips, psnr_full, psnr_region



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ckpt", default="runs/unet_rich/best.pt", help="segmenter used for the PREDICTED-mask rows")
    ap.add_argument("--lama_ft", default="runs/lama_ft/best.pt")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--dilate", type=int, default=5)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--save_n", type=int, default=12)
    ap.add_argument("--out", default="results")
    ap.add_argument("--no_lpips", action="store_true")
    a = ap.parse_args()

    data, out = Path(a.data), Path(a.out)
    (out / "samples_ft").mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg, mode = load_model(a.ckpt, device)
    zs, ft = Lama(None, device), Lama(a.lama_ft, device)
    lpips_val = None if a.no_lpips else make_lpips(device)
    if lpips_val is None:
        lpips_val = lambda x, y: None  # noqa: E731

    folder = data / "synthetic" / "test"
    idx = {r["tile_id"]: r for r in csv.DictReader(open(folder / "index.csv"))}
    ids = list(idx)[:a.n]
    rows = []
    for k, tid in enumerate(tqdm(ids, desc="evaluating")):
        clean = imread_rgb(data / idx[tid]["path"])
        damaged = imread_rgb(folder / f"{tid}_damaged.png")
        struct = imread_rgb(folder / f"{tid}_struct.png")
        gt_union = (imread_rgb(folder / f"{tid}_mask.png") > 127).any(axis=2).astype(np.uint8)
        m_or = dilate_mask(gt_union, a.dilate)
        m_pr = infill_mask(segment_image(seg, damaged, mode, device), a.thr, a.dilate)
        res = {"damaged (no restoration)": struct,
               "Telea, oracle mask": telea(struct, m_or),
               "Telea, predicted mask": telea(struct, m_pr),
               "LaMa zero-shot, oracle mask": composite(struct, zs(struct, m_or), m_or),
               "LaMa zero-shot, predicted mask": composite(struct, zs(struct, m_pr), m_pr),
               "LaMa fine-tuned, oracle mask": composite(struct, ft(struct, m_or), m_or),
               "LaMa fine-tuned, predicted mask": composite(struct, ft(struct, m_pr), m_pr)}
        for name, img in res.items():
            rows.append({"tile_id": tid, "method": name, "psnr_region": psnr_region(img, clean, m_or),
                         "psnr_full": psnr_full(img, clean),
                         "ssim_full": float(structural_similarity(img, clean, channel_axis=2)),
                         "lpips_full": lpips_val(img, clean)})
        if k < a.save_n:
            s = out / "samples_ft"
            imwrite_rgb(s / f"{tid}_clean.png", clean)
            imwrite_rgb(s / f"{tid}_damaged.png", struct)
            imwrite_gray(s / f"{tid}_mask.png", m_pr)
            imwrite_rgb(s / f"{tid}_telea.png", res["Telea, predicted mask"])
            imwrite_rgb(s / f"{tid}_lama.png", res["LaMa fine-tuned, predicted mask"])   # 07_make_figures shows "lama"
            imwrite_rgb(s / f"{tid}_lamazs.png", res["LaMa zero-shot, predicted mask"])

    with open(out / "inpaint_ft_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    lines = ["| Method | PSNR (damaged region) | PSNR (tile) | SSIM (tile) | LPIPS (tile, lower=better) |", "| --- | --- | --- | --- | --- |"]
    for m in dict.fromkeys(r["method"] for r in rows):
        rr = [r for r in rows if r["method"] == m]
        mean = lambda key: np.mean([r[key] for r in rr if r[key] is not None]) if any(r[key] is not None for r in rr) else float("nan")  # noqa: E731
        lines.append(f"| {m} | {mean('psnr_region'):.2f} | {mean('psnr_full'):.2f} | {mean('ssim_full'):.4f} | {mean('lpips_full'):.4f} |")
    (out / "inpaint_ft_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nsaved to {out}/ (inpaint_ft_metrics.csv, inpaint_ft_summary.md, samples_ft/)")


if __name__ == "__main__":
    main()
