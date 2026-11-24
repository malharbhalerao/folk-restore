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
# ---- version 2 (milestones 2-4)
python 03_train_unet.py --name unet_v2 --decoy_prob 0.4 --workers 3
python 09_finetune_lama.py --workers 3
python 10_eval_inpaint.py --ckpt runs/unet_v2/best.pt
python 11_train_pix2pix_defade.py --workers 3
python 11_train_pix2pix_defade.py --workers 3 --lambda_gan 0 --name defade_l1ref --out results_l1ref
python 12_eval_sr_uncertainty.py --ckpt runs/unet_v2/best.pt
python restore.py data/real_damaged -o results/final --ckpt runs/unet_v2/best.pt --sr 2
jupyter nbconvert --to notebook --execute validation.ipynb --output validation_executed.ipynb

# --- v3 (wall-mural damage) ---
python 02_make_synthetic.py --profile wall --scale_jitter 0.5 --photo 0.5 --out_name synthetic_wall --seed 321
python 03_train_unet.py --name unet_v3 --decoy_prob 0.4 --profile mixed --scale_jitter 0.5 --photo 0.5 --val_dir data/synthetic_wall/val
python 04_evaluate.py --ckpt runs/unet_v3/best.pt --test_dir data/synthetic_wall/test --out results_v3_wall --no_lama
python 13_diagnose_real.py data/real_damaged --ckpt runs/unet_v3/best.pt --out results/diagnose
