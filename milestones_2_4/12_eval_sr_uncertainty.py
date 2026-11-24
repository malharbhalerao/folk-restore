"""Step 12 (Milestone 4 results): does the uncertainty map mean anything, and does tiled super-resolution work?

A. Discriminator calibration: logits of the LaMa-fine-tuning critic on CLEAN tiles -> runs/<name>/disc_calib.json
B. Uncertainty validity on the synthetic TEST set (where the true restoration error is known):
     restore each damaged tile (predicted mask + LaMa), then compare the uncertainty of every repaired pixel with its true error
     |restored - clean|.  Reported per signal and for the fusion:  Spearman rho, AUROC for "pixel is badly wrong"
     (error > --err_thr), and the mean error of the most-trusted half vs the least-trusted half of repaired pixels.
C. Super-resolution: clean 512 tile -> 128 (area down-sampling) -> bicubic x4 vs Real-ESRGAN x4 (tiled), scored against the
     clean tile with PSNR / SSIM / LPIPS; plus tiled-vs-untiled difference (seams) and timing.

    python 12_eval_sr_uncertainty.py --lama_ft runs/lama_ft/best.pt --disc runs/lama_ft/disc.pt
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.stats import rankdata, spearmanr
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

import uncertainty as U
from common import dilate_mask, imread_rgb, imwrite_rgb, infill_mask, load_model, segment_image
from inpaint_lib import LamaJit, disc_logits, load_disc
from sr_lib import load_esrgan, sr_x4


def auroc(score, positive):
    """P(score of a random positive > score of a random negative), via ranks (Mann-Whitney)."""
    positive = positive.astype(bool)
    n1, n0 = positive.sum(), (~positive).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ckpt", default="runs/unet_rich/best.pt")
    ap.add_argument("--lama_ft", default="runs/lama_ft/best.pt")
    ap.add_argument("--disc", default="runs/lama_ft/disc.pt")
    ap.add_argument("--n", type=int, default=60, help="tiles for the uncertainty test")
    ap.add_argument("--n_sr", type=int, default=20, help="tiles for the super-resolution test")
    ap.add_argument("--n_calib", type=int, default=40, help="clean tiles for discriminator calibration (from the VAL split)")
    ap.add_argument("--err_thr", type=float, default=25.0, help="mean |RGB error| above which a repaired pixel counts as wrong")
    ap.add_argument("--sr_tile", type=int, default=64, help="LR tile size for the tiled SR test (small, to exercise tiling)")
    ap.add_argument("--sr_overlap", type=int, default=16)
    ap.add_argument("--out", default="results")
    ap.add_argument("--skip_sr", action="store_true")
    ap.add_argument("--no_lpips", action="store_true")
    a = ap.parse_args()

    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ A. calibrate the discriminator
    D, _ = load_disc(a.disc, device)
    vrows = [r for r in csv.DictReader(open(data / "manifest.csv")) if r["split"] == "val"]
    vrows = [vrows[i] for i in rng.permutation(len(vrows))[:a.n_calib]]
    maps = [disc_logits(D, imread_rgb(data / r["path"]), device) for r in tqdm(vrows, desc="calibrating D on clean tiles")]
    calib = U.calibrate_disc(maps)
    json.dump(calib, open(Path(a.disc).with_name("disc_calib.json"), "w"), indent=2)
    print("discriminator calibration on clean patches:", calib)

    # ------------------------------------------------------------------ B. uncertainty vs true error
    seg, mode = load_model(a.ckpt, device)
    lama = LamaJit(a.lama_ft, device)
    folder = data / "synthetic" / "test"
    idx = list(csv.DictReader(open(folder / "index.csv")))[:a.n]
    sig_names = ["entropy", "fill", "disagree", "disc", "fused"]
    pool = {k: [] for k in sig_names}
    errs, per_tile = [], []
    for r in tqdm(idx, desc="uncertainty test"):
        tid = r["tile_id"]
        clean = imread_rgb(data / r["path"])
        damaged = imread_rgb(folder / f"{tid}_damaged.png")
        struct = imread_rgb(folder / f"{tid}_struct.png")
        prob = segment_image(seg, damaged, mode, device)
        mask = infill_mask(prob, 0.5, 5)
        if mask.sum() == 0:
            continue
        variants = U.tta_inpaint(lama, struct, mask)
        restored = variants[0]
        from common import composite
        restored = composite(struct, restored, mask)
        sig = {"entropy": U.signal_entropy(prob), "fill": U.signal_fill(mask),
               "disagree": U.signal_disagree(variants, mask),
               "disc": U.signal_disc(disc_logits(D, restored, device), calib)}
        conf, unc = U.fuse(sig, mask)
        sig["fused"] = unc
        err = np.abs(restored.astype(np.float32) - clean.astype(np.float32)).mean(-1)
        sel = mask > 0
        errs.append(err[sel])
        for k in sig_names:
            pool[k].append(sig[k][sel])
        per_tile.append({"tile_id": tid, "hole_frac": float(sel.mean()), "mean_err_inside": float(err[sel].mean()),
                         "mean_unc_inside": float(unc[sel].mean())})
    if not errs:
        raise SystemExit("No tile had a predicted mask; nothing to evaluate.")
    err = np.concatenate(errs)
    wrong = err > a.err_thr
    print(f"\n{len(err):,} repaired pixels from {len(per_tile)} tiles; {100 * wrong.mean():.1f}% have error > {a.err_thr:.0f}")
    lines = ["| Signal | Spearman rho vs error | AUROC (error > %g) | mean error, most-trusted half | mean error, least-trusted half |" % a.err_thr,
             "| --- | --- | --- | --- | --- |"]
    for k in sig_names:
        s = np.concatenate(pool[k])
        rho = spearmanr(s, err)[0] if s.std() > 0 else float("nan")
        med = np.median(s)
        lo, hi = err[s <= med].mean(), err[s > med].mean()
        lines.append(f"| {k} | {rho:.3f} | {auroc(s, wrong):.3f} | {lo:.1f} | {hi:.1f} |")
    tile_rho = spearmanr([t["mean_unc_inside"] for t in per_tile], [t["mean_err_inside"] for t in per_tile])[0] if len(per_tile) > 2 else float("nan")
    lines.append(f"\nTile-level Spearman rho (mean uncertainty vs mean error across tiles): {tile_rho:.3f}")
    (out / "uncertainty_eval.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    with open(out / "uncertainty_per_tile.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_tile[0]))
        w.writeheader()
        w.writerows(per_tile)

    # ------------------------------------------------------------------ C. super-resolution
    if a.skip_sr:
        return
    net = load_esrgan(device)
    lp = None
    if not a.no_lpips:
        try:
            import lpips
            lp = lpips.LPIPS(net="alex", verbose=False).to(device)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] LPIPS unavailable ({e})")

    def lpips_val(x, y):
        if lp is None:
            return float("nan")
        t = lambda im: torch.from_numpy(im).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(device)  # noqa: E731
        with torch.no_grad():
            return float(lp(t(x), t(y)).item())

    rows = {m: {"psnr": [], "ssim": [], "lpips": []} for m in ("bicubic x4", "Real-ESRGAN x4 (tiled)")}
    seam, seam_edge, t_tiled, t_full = [], [], [], []
    (out / "sr_samples").mkdir(exist_ok=True)
    for k, r in enumerate(tqdm(idx[:a.n_sr], desc="super-resolution test")):
        hr = imread_rgb(data / r["path"])
        h, w = hr.shape[:2]
        lr = cv2.resize(hr, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        bic = cv2.resize(lr, (w // 4 * 4, h // 4 * 4), interpolation=cv2.INTER_CUBIC)
        t0 = time.time()
        sr_t = sr_x4(net, lr, tile=a.sr_tile, overlap=a.sr_overlap, device=device)
        t_tiled.append(time.time() - t0)
        t0 = time.time()
        sr_f = sr_x4(net, lr, tile=0, device=device)
        t_full.append(time.time() - t0)
        ref = hr[:h // 4 * 4, :w // 4 * 4]
        for name, im in (("bicubic x4", bic), ("Real-ESRGAN x4 (tiled)", sr_t)):
            rows[name]["psnr"].append(peak_signal_noise_ratio(ref, im))
            rows[name]["ssim"].append(structural_similarity(ref, im, channel_axis=2))
            rows[name]["lpips"].append(lpips_val(ref, im))
        d = np.abs(sr_t.astype(np.float32) - sr_f.astype(np.float32)).mean(-1)
        seam.append(float(d.mean()))
        edge = np.zeros(d.shape, bool)                       # pixels within 8 HR px of a tile boundary
        st = (a.sr_tile - a.sr_overlap) * 4
        for p in range(st, d.shape[0], st):
            edge[max(0, p - 8):p + 8] = True
        for p in range(st, d.shape[1], st):
            edge[:, max(0, p - 8):p + 8] = True
        seam_edge.append(float(d[edge].mean()) if edge.any() else float("nan"))
        if k < 4:
            imwrite_rgb(out / "sr_samples" / f"{r['tile_id']}_lr_bicubic.png", bic)
            imwrite_rgb(out / "sr_samples" / f"{r['tile_id']}_sr.png", sr_t)
            imwrite_rgb(out / "sr_samples" / f"{r['tile_id']}_clean.png", ref)
    lines = ["| Method (128 px -> 512 px) | PSNR dB | SSIM | LPIPS (lower = better) |", "| --- | --- | --- | --- |"]
    for m, v in rows.items():
        lines.append(f"| {m} | {np.mean(v['psnr']):.2f} | {np.mean(v['ssim']):.4f} | {np.nanmean(v['lpips']):.4f} |")
    lines.append(f"\nTiled (LR tile {a.sr_tile}, overlap {a.sr_overlap}) vs untiled output: mean abs difference {np.mean(seam):.3f} / 255 overall, "
                 f"{np.nanmean(seam_edge):.3f} / 255 within 8 px of tile boundaries.")
    lines.append(f"Time per tile: tiled {np.mean(t_tiled):.2f} s, untiled {np.mean(t_full):.2f} s on {device}.")
    (out / "sr_eval.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
