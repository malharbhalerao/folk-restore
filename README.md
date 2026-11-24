# folk-restore

Digital restoration of Indian folk paintings (Madhubani, Pattachitra, Warli, Kalamkari; also tested on Mandawa wall paintings).
CS F407 Artificial Intelligence course project, team MARS.

```
damaged image ─► M1 segment ─► M2 inpaint ─► M3 de-fade ─► M4 super-resolve ─► restored PNG
                  (CLAHE +        (Telea /      (L1 U-Net /    (Real-ESRGAN,        + confidence map
                   U-Net)          LaMa /        Pix2Pix +      tiled)               (4 signals)
                                   fine-tuned)   margin ref.)
```

> Quantitative results are on *simulated* damage. Results on real paintings are qualitative. Colours of web / phone photos are uncalibrated,
> so colour claims on real images cannot be verified.

## Layout
```
folk_restore/    package: config, io, degrade, segment, inpaint, colour, gan, sr, uncertainty, metrics
restore.py       integrated CLI: damaged painting -> restored PNG + confidence map
scripts/data/    collect, ingest, contact_sheet, prepare_tiles, make_synthetic, leakage_check
scripts/train/   segmenter, lama, defade, defade_gan
scripts/eval/    synthetic, segmenter, inpaint, defade_baseline, uncertainty_sr, diagnose_real
scripts/         make_figures.py, run_all.sh
notebooks/       validation.ipynb (loss curves, patch and augmentation stats), kaggle_runner.ipynb
tests/           CPU smoke tests          docs/  dataset-collection notes
data/ runs/ results/ figures/   generated, not versioned (data/sources.csv is tracked for licences)
```

## Setup
```
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision                    # pick the wheel for your CUDA version
pip install -r requirements.txt && pip install -e .
pip install simple-lama-inpainting --no-deps     # pretrained Big-LaMa weights
```
Run everything from the repository root. Weights are downloaded on first use (ImageNet ResNet-34, Big-LaMa, Real-ESRGAN, LPIPS).

## Use
```
python restore.py painting.jpg -o out/ --ckpt runs/unet_v3/best.pt --sr 2
python restore.py folder/ -o out/ --margin x0,y0,x1,y1       # optional: an un-faded strip for the colour model
```
Outputs per image: `_restored.png`, `_confidence.npy` (1 = trusted, 0 = speculative), `_confidence.png`, `_review.png`, `_mask.png`.
Each stage falls back (fine-tuned LaMa → pretrained LaMa → Telea; Pix2Pix → L1 de-fader → skipped) if its checkpoint is missing.

## Reproduce
Put images in `data/raw/<style>/` (see `docs/dataset_collection.md`, licences in `data/sources.csv`) and run `bash scripts/run_all.sh`.
Individual steps and flags are documented in each script's docstring.

## Milestones
| | Method | Script |
|---|---|---|
| M1 | CLAHE + black-hat features, U-Net (ResNet-34); hard-negative design lines, plaster-wall damage | `train/segmenter.py` |
| M2 | Big-LaMa fine-tuned on our tiles (L1 + VGG + PatchGAN), vs zero-shot LaMa and Telea | `train/lama.py`, `eval/inpaint.py` |
| M3 | Pix2Pix de-fader with protected-margin colour reference; L1-only ablation; Reinhard negative baseline | `train/defade_gan.py` |
| M4 | tiled Real-ESRGAN; 4-signal confidence map (entropy, hole depth, inpainter disagreement, discriminator) | `eval/uncertainty_sr.py` |

The professor's Mandawa test set is never trained on or tuned on, and is not redistributed.
