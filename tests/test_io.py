"""Tests for image resizing without model checkpoints."""
import numpy as np
import pytest

from folk_restore.io import resize_long_side


@pytest.mark.parametrize("shape, expected", [
    ((1, 1000, 3), (1, 100, 3)),
    ((1000, 1, 3), (100, 1, 3)),
    ((200, 400, 3), (50, 100, 3)),
])
def test_resize_long_side_keeps_dimensions_positive(shape, expected):
    img = np.full(shape, 127, dtype=np.uint8)
    resized = resize_long_side(img, 100)
    assert resized.shape == expected
    assert resized.dtype == img.dtype
    np.testing.assert_array_equal(resized, np.full(expected, 127, dtype=np.uint8))


@pytest.mark.parametrize("max_side", [0, -1])
def test_resize_long_side_rejects_nonpositive_limit(max_side):
    with pytest.raises(ValueError, match="max_side must be positive"):
        resize_long_side(np.zeros((10, 20, 3), dtype=np.uint8), max_side)


def test_resize_long_side_leaves_small_images_unchanged():
    img = np.zeros((50, 100, 3), dtype=np.uint8)
    assert resize_long_side(img, 100) is img
    assert resize_long_side(img, 200) is img
