"""Fast CPU smoke tests: python -m pytest tests"""
import numpy as np

from folk_restore import CLASSES, composite, dice_np, infill_mask, telea
from folk_restore.degrade import degrade, degrade_v2


def _tile(seed=0):
    r = np.random.default_rng(seed)
    return (r.random((128, 128, 3)) * 200 + 30).astype(np.uint8)


def test_degrade_shapes_and_masks():
    d = degrade(_tile(), np.random.default_rng(1))
    assert d["image"].shape == (128, 128, 3)
    assert set(d["masks"]) == set(CLASSES)


def test_degrade_v2_clean_negatives_have_empty_masks():
    rng = np.random.default_rng(0)
    seen_clean = False
    for _ in range(40):
        d = degrade_v2(_tile(), rng, p_clean=0.5)
        if not any(m.any() for m in d["masks"].values()):
            seen_clean = True
    assert seen_clean


def test_infill_and_inpaint_roundtrip():
    prob = np.zeros((128, 128, 3), np.float32)
    prob[40:60, 40:60, 1] = 0.9
    m = infill_mask(prob, classes=["loss"])
    assert m[50, 50] == 255 and infill_mask(prob, classes=["crack"]).sum() == 0
    img = _tile()
    out = composite(img, telea(img, m), m)
    assert (out[m == 0] == img[m == 0]).sum() > 0.9 * (m == 0).sum() * 3 * 0.99
    assert dice_np(m > 0, m > 0) > 0.99
