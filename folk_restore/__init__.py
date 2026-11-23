"""folk_restore: digital restoration of Indian folk paintings."""
from .colour import (build_defade_model, defade_forward, defade_image, lab_stats, load_defade, reinhard)
from .config import CLASSES, IMG_EXTS, SOURCE_FIELDS, STYLES
from .inpaint import Lama, composite, telea, try_lama
from .io import (append_sources, imread_rgb, imwrite_gray, imwrite_rgb, list_images, resize_long_side)
from .segment import (build_model, dice_np, dilate_mask, features, infill_mask, load_model, segment_image)
from .sr import upscale
