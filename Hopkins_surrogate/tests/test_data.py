from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hopkins_surrogate.data import (base_to_index, load_aerial_binary,
                                    load_cache_meta, load_layout_mask,
                                    load_splits, load_target,
                                    load_target_native, make_group_splits,
                                    assert_no_leakage)
from hopkins_surrogate.metrics import binary_iou

DATASET = Path("E:/diffusio4ilt_dataset")
HAS_DATA = (DATASET / "layout/png/layout_00001.png").is_file()


def _nearest_upsample(binary: np.ndarray, size: int) -> np.ndarray:
    img = Image.fromarray((binary.astype(np.uint8) * 255), mode="L")
    img = img.resize((size, size), resample=Image.Resampling.NEAREST)
    return np.asarray(img) >= 128


def test_base_to_index():
    assert base_to_index("layout_00001") == 0
    assert base_to_index("layout_10000") == 9999
    with pytest.raises(ValueError):
        base_to_index("aerial_image_00001")


def test_group_splits_synthetic():
    records = []
    for i in range(200):
        records.append({
            "base": f"layout_{i + 1:05d}",
            "config_hash": "h1" if i < 20 else "h2",
            "signature": f"sig_{i % 20}",
        })
    splits = make_group_splits(records, range(200), seed=0)
    assert_no_leakage(splits)
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == 200
    assert len(splits["val"]) > 0 and len(splits["test"]) > 0
    # no signature group crosses splits
    for name in ("train", "val", "test"):
        assert len(set(splits["signature_counts"][name])) >= 1


FULL_CACHE = Path("E:/hopkins_surrogate/cache_v1")


@pytest.mark.skipif(not (FULL_CACHE / "cache_meta.json").is_file(),
                    reason="full cache not available")
def test_full_cache_metadata():
    meta = load_cache_meta(FULL_CACHE)
    splits = load_splits(FULL_CACHE)
    assert meta["n"] == 10000
    assert meta["shape"] == [800, 800]
    assert splits["counts"] == {"train": 8000, "val": 1000, "test": 1000}
    masks = np.load(FULL_CACHE / "masks.npy", mmap_mode="r")
    assert masks.shape == (10000, 800, 800)


@pytest.mark.skipif(not HAS_DATA, reason="dataset not available")
def test_dat_png_alignment_sample1():
    dat = DATASET / "aerial_image/dat/aerial_image_00001.dat"
    native = load_target_native(dat)
    assert native.shape == (320, 320)
    target = load_target(dat, size=800)
    assert target.shape == (800, 800)
    assert np.isfinite(target).all()
    layout = load_layout_mask(DATASET / "layout/png/layout_00001.png")
    assert layout.shape == (800, 800)
    assert set(np.unique(layout)).issubset({0, 1})
    png = load_aerial_binary(DATASET / "aerial_image/png/aerial_image_00001.png")
    up = _nearest_upsample(native > 0.346585, 800)
    assert binary_iou(up, png > 0) > 0.999
