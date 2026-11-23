"""generate the FIXED synthetic val/test sets (training data is generated on the fly in step 3).

For every sampled tile writes into data/synthetic/<split>/ :
    <id>_damaged.png  damaged + faded      (segmenter input)
    <id>_struct.png   damaged, not faded   (inpainting evaluation input)
    <id>_faded.png    clean tile + fade    (de-fading evaluation input)
    <id>_mask.png     R=crack G=loss B=stain (0/255)
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm

from folk_restore import CLASSES, imread_rgb, imwrite_gray, imwrite_rgb
from folk_restore.degrade import degrade, degrade_v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--hard", action="store_true",
                    help="also write synthetic/test_hard: distractor design strokes, 25%% damage-free tiles, photometric variation")
    a = ap.parse_args()

    data = Path(a.data)
    rows = list(csv.DictReader(open(data / "manifest.csv")))
    rng = np.random.default_rng(a.seed)
    jobs = [("val", "val", a.n_val, False), ("test", "test", a.n_test, False)]
    if a.hard:
        jobs.append(("test_hard", "test", a.n_test, True))
    for name, split, n, hard in jobs:
        pool = [r for r in rows if r["split"] == split]
        if not pool:
            print(f"[warn] no tiles in split '{split}'")
            continue
        idx = rng.permutation(len(pool))[:n]
        out = data / "synthetic" / name
        index = []
        for i in tqdm(idx, desc=f"synthetic {name}"):
            r = pool[int(i)]
            clean = imread_rgb(data / r["path"])
            d = degrade_v2(clean, rng, fade_prob=1.0, p_clean=0.25, p_distract=0.8) if hard else degrade(clean, rng, fade_prob=1.0)
            if hard:
                imwrite_rgb(out / f"{r['tile_id']}_clean.png", d["clean_aug"])
            tid = r["tile_id"]
            imwrite_rgb(out / f"{tid}_damaged.png", d["image"])
            imwrite_rgb(out / f"{tid}_struct.png", d["image_struct"])
            imwrite_rgb(out / f"{tid}_faded.png", d["faded_only"])
            imwrite_rgb(out / f"{tid}_mask.png", np.stack([d["masks"][c] * 255 for c in CLASSES], axis=2))
            index.append({"tile_id": tid, "path": r["path"], "style": r["style"]})
        with open(out / "index.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["tile_id", "path", "style"])
            w.writeheader()
            w.writerows(index)
        print(f"{name}: {len(index)} synthetic tiles -> {out}")


if __name__ == "__main__":
    main()
