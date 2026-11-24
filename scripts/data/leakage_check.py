"""Leakage check: near-duplicate (dHash, Hamming <= thr) test images vs. the training/demo data.

    python scripts/data/leakage_check.py --test_dir ../Dataset --against data/raw data/real_damaged
"""
import argparse

import cv2
import numpy as np

from folk_restore import imread_rgb, list_images


def dhash(rgb):
    g = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--against", nargs="+", default=["data/raw", "data/real_damaged"])
    ap.add_argument("--thr", type=int, default=6)
    a = ap.parse_args()
    ref = [(p, dhash(imread_rgb(p))) for d in a.against for p in list_images(d)]
    hits = 0
    for p in list_images(a.test_dir):
        h = dhash(imread_rgb(p))
        best = min(ref, key=lambda r: np.count_nonzero(r[1] != h))
        dist = int(np.count_nonzero(best[1] != h))
        if dist <= a.thr:
            hits += 1
            print(f"OVERLAP {p.name} ~ {best[0]} (hamming {dist})")
    print(f"checked {len(list_images(a.test_dir))} test images against {len(ref)} reference images: {hits} overlaps")


if __name__ == "__main__":
    main()
