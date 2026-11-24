#!/usr/bin/env bash
# Full pipeline from the repo root, after data/raw/<style>/*.jpg and data/real_damaged/ are in place.
# ~2.5 h on a 6 GB RTX 3050. Every step is seeded and writes under runs/, results/.
set -e
python scripts/data/prepare_tiles.py --stride 256 --max_tiles 80 --min_side 500 --test_frac 0.15
python scripts/data/make_synthetic.py                                                       # paper-style val/test
python scripts/data/make_synthetic.py --profile wall --scale_jitter 0.5 --photo 0.5 --out_name synthetic_wall --seed 321

# M1: segmenter. v1 = midsem baseline, v3 = hard negatives + wall damage + scale/photo augmentation
python scripts/train/segmenter.py --name unet_rich --workers 6                              # v1 (midsem recipe)
python scripts/train/segmenter.py --name unet_v3 --decoy_prob 0.4 --profile mixed --scale_jitter 0.5 --photo 0.5 \
       --val_dir data/synthetic_wall/val --epochs 20 --workers 6
python scripts/train/segmenter.py --name unet_rgb --mode rgb --workers 6                    # CLAHE ablation
python scripts/train/segmenter.py --mode rgb --name unet_v3_rgb --decoy_prob 0.4 --profile mixed --scale_jitter 0.5 --photo 0.5 \
       --val_dir data/synthetic_wall/val --epochs 20 --workers 6
python scripts/eval/segmenter.py --ckpts runs/unet_rich/best.pt runs/unet_v3/best.pt

# M2: LaMa fine-tune, then inpainting comparison
python scripts/train/lama.py --epochs 8 --samples_per_epoch 800
python scripts/eval/inpaint.py --ckpt runs/unet_v3/best.pt --n 100
python scripts/eval/inpaint.py --ckpt runs/unet_v3/best.pt --n 50 --diffusion --tag inpaint_diffusion

# M3: colour
python scripts/eval/defade_baseline.py                                                      # Reinhard (negative baseline)
python scripts/train/defade.py --workers 6                                                  # L1 U-Net
python scripts/train/defade_gan.py --workers 6                                              # Pix2Pix + margin reference
python scripts/train/defade_gan.py --name defade_nogan --lambda_gan 0 --workers 6           # ablation

# M4: super-resolution + confidence map
python scripts/eval/uncertainty_sr.py --ckpt runs/unet_v3/best.pt

# integrated CLI on a folder of real paintings (the professor's Mandawa set was run this way)
python restore.py data/real_damaged -o results/restored --sr 2
python scripts/eval/audit_make.py --images data/real_damaged --out audit   # human-audit sheets
