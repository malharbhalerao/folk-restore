"""Fast CPU smoke tests: python -m pytest tests"""
import numpy as np

from folk_restore import CLASSES, composite, dice_np, infill_mask, telea
from folk_restore import uncertainty as U
from folk_restore.degrade import degrade


def _tile(seed=0):
    r = np.random.default_rng(seed)
    return (r.random((128, 128, 3)) * 200 + 30).astype(np.uint8)


def test_degrade_shapes_and_masks():
    d = degrade(_tile(), np.random.default_rng(1))
    assert d["image"].shape == (128, 128, 3)
    assert set(d["masks"]) == set(CLASSES)


def test_degrade_wall_profile_and_decoys_keep_masks_consistent():
    d = degrade(_tile(), np.random.default_rng(2), profile="wall", decoy_prob=1.0, scale_jitter=1.0, photo=1.0)
    assert d["image"].shape == (128, 128, 3) and d["clean"].shape == (128, 128, 3)
    assert all(m.shape == (128, 128) for m in d["masks"].values())


def test_confidence_fusion_range():
    prob = np.zeros((64, 64, 3), np.float32)
    mask = np.zeros((64, 64), np.uint8)
    mask[20:40, 20:40] = 255
    conf, unc = U.fuse({"entropy": U.signal_entropy(prob), "fill": U.signal_fill(mask)}, mask)
    assert conf.min() >= 0 and conf.max() <= 1 and conf[30, 30] < conf[2, 2]


def test_infill_and_inpaint_roundtrip():
    prob = np.zeros((128, 128, 3), np.float32)
    prob[40:60, 40:60, 1] = 0.9
    m = infill_mask(prob, classes=["loss"])
    assert m[50, 50] == 255 and infill_mask(prob, classes=["crack"]).sum() == 0
    img = _tile()
    out = composite(img, telea(img, m), m)
    assert (out[m == 0] == img[m == 0]).sum() > 0.9 * (m == 0).sum() * 3 * 0.99
    assert dice_np(m > 0, m > 0) > 0.99
