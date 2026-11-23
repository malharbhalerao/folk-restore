# folk-restore

Digital restoration of Indian folk paintings (Madhubani, Pattachitra, Warli, Kalamkari; tested on Mandawa wall paintings).
CS F407 Artificial Intelligence course project, team MARS.

Pipeline: **damage segmentation** (CLAHE + black-hat + U-Net) → **inpainting** (Telea / pretrained LaMa / diffusion) →
**colour de-fading** (learned LAB residual, L1 and Pix2Pix variants) → **super-resolution** (Real-ESRGAN) → **confidence map**.

> Quantitative results are on *simulated* damage. Results on real paintings are qualitative. LaMa is pretrained, not fine-tuned.

## Layout
```
folk_restore/      importable package: config, io, degrade (damage simulator), segment, inpaint, colour, sr
scripts/data/      collect, ingest, contact_sheet, prepare_tiles, make_synthetic
scripts/train/     segmenter, defade
scripts/eval/      synthetic, segmenter, defade_baseline
scripts/           restore_real.py, make_figures.py, run_all.sh
tests/             CPU smoke tests
docs/              kaggle_notes.md (dataset collection notes from the midsem)
data/ runs/ results/ figures/   generated, not versioned (data/sources.csv is tracked for licences)
```

## Setup
```
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision                    # choose the wheel for your CUDA version
pip install -r requirements.txt && pip install -e .
pip install simple-lama-inpainting --no-deps     # pretrained LaMa
```
Run every script from the repository root.

## Reproduce
Put images in `data/raw/<style>/` and a held-out demo set in `data/real_damaged/` (see `docs/kaggle_notes.md` and
`data/sources.csv`), then `bash scripts/run_all.sh`. Model weights are not in git; see Releases.

Data licences are in `data/sources.csv`. The professor's Mandawa test set is never trained on and is not redistributed.
