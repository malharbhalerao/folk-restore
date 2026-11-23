#!/usr/bin/env bash
# Reproduce the v1 (midsem) pipeline from the repo root, after data/raw/<style>/*.jpg and data/real_damaged/ are in place.
set -e
python scripts/data/prepare_tiles.py --stride 256 --max_tiles 80 --min_side 500 --test_frac 0.15
python scripts/data/make_synthetic.py --hard
python scripts/train/segmenter.py --name unet_rich --workers 3                   # v1 segmenter
python scripts/eval/synthetic.py --ckpt runs/unet_rich/best.pt
python scripts/eval/defade_baseline.py
python scripts/train/defade.py --workers 3
python scripts/restore_real.py --input_dir data/real_damaged --defade_model runs/defade/best.pt
