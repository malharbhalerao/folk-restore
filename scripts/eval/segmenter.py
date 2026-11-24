"""segmenter comparison across checkpoints on synthetic test sets (paper-style and wall-style).

Reports Dice per class / union (on tiles with damage) and the false-alarm rate on damage-free tiles
(mean fraction of pixels flagged; this is the metric that tracks 'design lines erased as damage').
Bootstrap 95% CIs over tiles.

    python scripts/eval/segmenter.py --ckpts runs/unet_rich/best.pt runs/unet_v2_s0/best.pt --sets synthetic/test synthetic_wall/test
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from folk_restore import CLASSES, dice_np, imread_rgb, infill_mask, load_model, segment_image


def boot(x, n=1000, seed=0):
    x = np.asarray(x, float)
    r = np.random.default_rng(seed)
    m = [x[r.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--sets", nargs="+", default=["synthetic/test", "synthetic_wall/test"], help="folders under --data")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--out", default="results/seg_compare.json")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = {}
    for ck in a.ckpts:
        model, mode = load_model(ck, device)
        for s in a.sets:
            folder = Path(a.data) / s
            ids = [r["tile_id"] for r in csv.DictReader(open(folder / "index.csv"))][:a.n]
            dice = {c: [] for c in CLASSES + ["union"]}
            fa = []
            for tid in tqdm(ids, desc=f"{Path(ck).parent.name}:{s}"):
                img = imread_rgb(folder / f"{tid}_damaged.png")
                gt = imread_rgb(folder / f"{tid}_mask.png") > 127
                prob = segment_image(model, img, mode, device)
                pred = prob > a.thr
                if gt.any():
                    for i, c in enumerate(CLASSES):
                        if gt[..., i].any():
                            dice[c].append(dice_np(pred[..., i], gt[..., i]))
                    dice["union"].append(dice_np(pred.any(2), gt.any(2)))
                else:
                    fa.append(float((infill_mask(prob, a.thr, 1, 20) > 0).mean()))
            r = {k: boot(v) for k, v in dice.items() if v}
            if fa:
                r["false_alarm_area_on_clean_tiles"] = boot(fa)
                r["n_clean"] = len(fa)
            res[f"{ck}|{s}"] = r
            print(ck, s, json.dumps({k: v for k, v in r.items()}, default=float))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
