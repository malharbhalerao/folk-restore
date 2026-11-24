#!/usr/bin/env python
"""Integrated restoration pipeline (deliverable 1): deteriorated painting in -> restored high-resolution PNG + confidence map out.

    python restore.py painting.jpg -o out/                      # one file
    python restore.py data/real_damaged -o out/ --sr 2          # a whole folder
    python restore.py painting.jpg -o out/ --margin 0,0,60,900  # x0,y0,x1,y1 of an UN-faded strip (protected margin)

Stages (each one falls back gracefully if its model file is missing, and says so):
  1 damage segmentation   runs/unet_v3/best.pt (or --ckpt)                  -> binary infill mask
  2 inpainting            runs/lama_ft/best.pt (fine-tuned LaMa) else pretrained LaMa else OpenCV Telea
  3 colour de-fading      runs/defade_nogan/best.pt when --margin is given (margin-reference model), else runs/defade/best.pt (L1);
                          runs/defade_gan/best.pt (Pix2Pix) only with --pix2pix; skipped if no model exists
  4 super-resolution      Real-ESRGAN x4 (tiled), resampled to --sr (0 = off)
  5 confidence map        entropy / hole-depth / inpainting-disagreement / discriminator signals, fused

Outputs per image <name> in --out:
  <name>_restored.png          final image (super-resolved if --sr)
  <name>_confidence.npy        float16 matrix, same size as the final image, 1 = trusted, 0 = speculative
  <name>_confidence.png        heat-map of the same
  <name>_review.png            restored image with low-confidence areas tinted red (for the conservator)
  <name>_mask.png              the damage mask that was inpainted
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from folk_restore import uncertainty as U
from folk_restore import (composite, defade_image, imread_rgb, imwrite_gray, imwrite_rgb, infill_mask, list_images, load_defade,
                    load_model, resize_long_side, segment_image, telea)
from folk_restore.colour import defade_ref_image, load_g, ref_stats
from folk_restore.gan import disc_logits, load_disc
from folk_restore.inpaint import Lama
from folk_restore.sr import load_esrgan, super_resolve


def parse_margin(s, img):
    if not s:
        return None
    x0, y0, x1, y1 = [int(v) for v in s.split(",")]
    h, w = img.shape[:2]
    region = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if region.size == 0:
        raise SystemExit(f"--margin {s} is outside the image ({w}x{h})")
    return ref_stats(region)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="image file or folder")
    ap.add_argument("-o", "--out", default="restored")
    ap.add_argument("--ckpt", default="runs/unet_v3/best.pt", help="segmenter")
    ap.add_argument("--lama_ft", default="runs/lama_ft/best.pt")
    ap.add_argument("--disc", default="runs/lama_ft/disc.pt")
    ap.add_argument("--defade_ref", default="runs/defade_nogan/best.pt", help="margin-reference de-fader (used with --margin)")
    ap.add_argument("--defade_gan", default="runs/defade_gan/best.pt", help="Pix2Pix de-fader (used with --pix2pix; it scored worse than L1)")
    ap.add_argument("--pix2pix", action="store_true")
    ap.add_argument("--defade_l1", default="runs/defade/best.pt")
    ap.add_argument("--defade_strength", type=float, default=0.7, help="0 = no colour change, 1 = full correction")
    ap.add_argument("--no_defade", action="store_true")
    ap.add_argument("--margin", default=None, help="x0,y0,x1,y1 (in the ORIGINAL image's pixels) of an un-faded region")
    ap.add_argument("--max_side", type=int, default=1536, help="working resolution (long side) before super-resolution")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--dilate", type=int, default=7)
    ap.add_argument("--sr", type=int, default=2, choices=[0, 2, 4], help="super-resolution factor, 0 = off")
    ap.add_argument("--sr_tile", type=int, default=256, help="LR tile size; lower it if the GPU runs out of memory")
    ap.add_argument("--sr_overlap", type=int, default=32)
    ap.add_argument("--fusion", default="runs/lama_ft/fusion_weights.json", help="fitted confidence-fusion weights (from scripts/eval/uncertainty_sr.py)")
    ap.add_argument("--no_confidence", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    src = Path(a.input)
    files = [src] if src.is_file() else list_images(src)
    if a.limit:
        files = files[:a.limit]
    if not files:
        raise SystemExit(f"No images found at {src}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    def have(p):
        return p and Path(p).exists()

    if not have(a.ckpt):
        raise SystemExit(f"Segmenter checkpoint {a.ckpt} not found. Train it with scripts/train/segmenter.py first.")
    seg, mode = load_model(a.ckpt, device)

    inpaint, inpaint_name = None, "OpenCV Telea"
    try:
        if have(a.lama_ft):
            inpaint, inpaint_name = Lama(a.lama_ft, device), "LaMa (fine-tuned)"
        else:
            inpaint, inpaint_name = Lama(None, device), "LaMa (pretrained, zero-shot)"
    except Exception as e:  # noqa: BLE001
        print(f"[warn] LaMa unavailable ({type(e).__name__}: {e}); using Telea")
    print("inpainting:", inpaint_name)

    gan = l1 = None
    if not a.no_defade:
        ref_path = a.defade_gan if a.pix2pix else (a.defade_ref if a.margin else None)
        if have(ref_path):
            gan = load_g(ref_path, device)
            print(f"de-fading: {ref_path}" + (" with margin reference" if a.margin else " (no margin given)"))
        elif have(a.defade_l1):
            l1 = load_defade(a.defade_l1, device)
            print("de-fading: L1 U-Net; margin reference unused")
        else:
            print("[warn] no de-fading model found; colour step skipped")

    disc = calib = None
    if not a.no_confidence and have(a.disc):
        disc, _ = load_disc(a.disc, device)
        cp = Path(a.disc).with_name("disc_calib.json")
        calib = json.load(open(cp)) if cp.exists() else None
        if calib is None:
            print("[warn] no disc_calib.json (run scripts/eval/uncertainty_sr.py once); the discriminator signal is skipped")
            disc = None
    sr_net = load_esrgan(device) if a.sr else None

    for p in files:
        name = p.stem
        orig = imread_rgb(p)
        img = resize_long_side(orig, a.max_side)
        scale = img.shape[1] / orig.shape[1]
        stats = None
        if a.margin:
            m = [int(float(v) * scale) for v in a.margin.split(",")]
            stats = parse_margin(",".join(map(str, m)), img)   # taken from the input, before any inpainting

        prob = segment_image(seg, img, mode, device)
        mask = infill_mask(prob, a.thr, a.dilate)
        variants = None
        if inpaint is None:
            filled = telea(img, mask)
        else:
            if a.no_confidence or mask.sum() == 0:
                variants = [inpaint(img, mask)]
            else:
                variants = U.tta_inpaint(inpaint, img, mask)
            filled = composite(img, variants[0], mask)

        final = filled
        if gan is not None:
            final = defade_ref_image(gan, final, device, stats, a.defade_strength)
        elif l1 is not None:
            final = defade_image(l1, final, device, a.defade_strength)

        conf = None
        if not a.no_confidence:
            sig = {"entropy": U.signal_entropy(prob), "fill": U.signal_fill(mask)}
            if variants is not None and len(variants) > 1:
                sig["disagree"] = U.signal_disagree(variants, mask)
            if disc is not None:
                sig["disc"] = U.signal_disc(disc_logits(disc, filled, device), calib)
            fit = U.load_fusion(a.fusion)
            conf, _ = U.fuse(sig, mask, weights=fit["weights"], scale=fit["scale"]) if fit else U.fuse(sig, mask)

        if a.sr:
            final = super_resolve(sr_net, final, a.sr, a.sr_tile, a.sr_overlap)
            if conf is not None:
                conf = U.upsample_conf(conf, final.shape[:2])
        imwrite_rgb(out / f"{name}_restored.png", final)
        imwrite_gray(out / f"{name}_mask.png", mask)
        if conf is not None:
            np.save(out / f"{name}_confidence.npy", conf.astype(np.float16))
            imwrite_rgb(out / f"{name}_confidence.png", U.heatmap(conf))
            imwrite_rgb(out / f"{name}_review.png", U.overlay(final, conf))
        print(f"{name}: repaired {100 * (mask > 0).mean():.1f}% of the surface"
              + (f", mean confidence {conf.mean():.3f}" if conf is not None else "") + f" -> {final.shape[1]}x{final.shape[0]}")
    print(f"\nsaved to {out}/")


if __name__ == "__main__":
    main()
