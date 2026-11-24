"""folk_restore: digital restoration of Indian folk paintings.

config      constants                      segment     M1: features, U-Net, mask post-processing
io          image / provenance I/O         inpaint     M2: Telea, Big-LaMa (pretrained or fine-tuned), masks
degrade     synthetic damage simulator     colour      M3: Reinhard baseline, L1 de-fader, Pix2Pix + margin reference
gan         PatchGAN critic, VGG loss      sr          M4: tiled Real-ESRGAN
uncertainty M4: confidence map             metrics     PSNR / LPIPS / dE00
"""
from .colour import (DEFADE_NORM, DEFADE_SCALE, build_defade_model, defade_forward, defade_image, lab_stats, load_defade,
                     reinhard)
from .config import CLASSES, IMG_EXTS, SOURCE_FIELDS, STYLES
from .inpaint import Lama, composite, telea, try_lama
from .io import append_sources, imread_rgb, imwrite_gray, imwrite_rgb, list_images, resize_long_side
from .segment import build_model, dice_np, dilate_mask, features, infill_mask, load_model, segment_image
from .sr import super_resolve
