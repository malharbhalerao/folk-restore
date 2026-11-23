"""Step 6 (Milestone 3 baseline): Reinhard colour transfer toward the training-set palette.

Builds data/ref_stats.npz (mean/std of LAB over clean training tiles) and measures
CIEDE2000 on the synthetic test set: faded tile vs. Reinhard(faded tile), both compared to the clean tile.
Lower dE00 = closer to the original colours. Several strengths are tried so you can pick the best.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from tqdm import tqdm

from folk_restore import imread_rgb, imwrite_rgb, lab_stats, reinhard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--n_ref", type=int, default=300)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = [r for r in csv.DictReader(open(data / "manifest.csv")) if r["split"] == "train"]
    rng = np.random.default_rng(0)
    pick = rng.permutation(len(rows))[:a.n_ref]
    stats = [lab_stats(imread_rgb(data / rows[int(i)]["path"])) for i in tqdm(pick, desc="reference palette")]
    mean = np.mean([s[0] for s in stats], axis=0)
    std = np.mean([s[1] for s in stats], axis=0)
    np.savez(data / "ref_stats.npz", mean=mean, std=std)
    print(f"reference LAB mean={np.round(mean, 1)} std={np.round(std, 1)} -> {data / 'ref_stats.npz'}")

    folder = data / "synthetic" / "test"
    idx = list(csv.DictReader(open(folder / "index.csv")))[:a.n]
    strengths = [0.0, 0.4, 0.6, 0.8, 1.0]
    de = {s: [] for s in strengths}
    for k, r in enumerate(tqdm(idx, desc="de-fading eval")):
        clean = imread_rgb(data / r["path"])
        faded = imread_rgb(folder / f"{r['tile_id']}_faded.png")
        lab_c = rgb2lab(clean)
        for s in strengths:
            img = faded if s == 0 else reinhard(faded, mean, std, s)
            de[s].append(float(deltaE_ciede2000(lab_c, rgb2lab(img)).mean()))
            if k < 8 and s in (0.0, 0.6):
                imwrite_rgb(out / "defade_samples" / f"{r['tile_id']}_{'faded' if s == 0 else 'defade'}.png", img)
        if k < 8:
            imwrite_rgb(out / "defade_samples" / f"{r['tile_id']}_cleanref.png", clean)

    lines = ["| Method | mean dE00 vs clean (lower = better) |", "| --- | --- |"]
    for s in strengths:
        name = "faded input (no correction)" if s == 0 else f"Reinhard, strength {s}"
        lines.append(f"| {name} | {np.mean(de[s]):.2f} |")
    (out / "defade_metrics.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print("\nNote: global Reinhard pulls every image toward the average palette; it is a baseline, not the final method.")


if __name__ == "__main__":
    main()
