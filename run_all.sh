#!/usr/bin/env bash
# After the dataset is in data/raw (and damaged demo images moved to data/real_damaged): ~1-1.5 h on a T4.
set -e
python 01_prepare_tiles.py
python 02_make_synthetic.py
python 03_train_unet.py --name unet_rich --workers 3
# python 03_train_unet.py --mode rgb --name unet_rgb --workers 3      # optional CLAHE ablation
python 04_evaluate.py --ckpt runs/unet_rich/best.pt
python 06_defade_eval.py
python 08_train_defade.py --workers 3
python 05_restore_real.py --input_dir data/real_damaged --defade_model runs/defade/best.pt
python 07_make_figures.py --dir results/samples --out figures/synthetic
python 07_make_figures.py --dir results/real --out figures/real
python 07_make_figures.py --dir results/defade_learned_samples --out figures/defade
