import json
from math import ceil
from pathlib import Path

import numpy as np
from ilt import layouts
from PIL import Image


DEFAULT_DATASET_MASK = "layout_00001"
M1_TEST3 = (
    Path(layouts.__file__).resolve().parents[2] / "benchmarks" / "ICCAD2013" / "M1_test3.glp"
)
NATIVE_FIELD_NM = 1024


def dataset_layout_target(grid_size: int, pixel_size_nm: float, *, root: Path, sample=DEFAULT_DATASET_MASK):
    """Load the dataset's native binary mask as an opaque-feature target."""
    if not sample or Path(sample).name != sample or "/" in sample or "\\" in sample:
        raise ValueError("Dataset sample must be a filename stem, e.g. layout_00001")
    path = root / "png" / f"{sample}.png"
    sidecar = root / "json" / f"{sample}.json"
    png_metadata = json.loads(sidecar.read_text(encoding="utf-8"))["png"]
    source_pitch_nm = 1.0 / png_metadata["px_per_nm"]
    expected_size = (grid_size, grid_size)
    if tuple(png_metadata["size_px"]) != expected_size or not np.isclose(source_pitch_nm, pixel_size_nm):
        raise ValueError(f"Dataset target {path} must match {expected_size} at {pixel_size_nm} nm/px")
    if (png_metadata["foreground"], png_metadata["background"]) != ("black", "white"):
        raise ValueError(f"Expected black features on white background: {path}")

    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L"))
    if pixels.shape != expected_size or not np.all((pixels == 0) | (pixels == 255)):
        raise ValueError(f"Expected a binary {expected_size} target: {path}")

    metadata = {
        "path": str(path),
        "metadata_path": str(sidecar),
        "sample": sample,
        "source_pixel_size_nm": source_pitch_nm,
        "source_convention": "black_feature_on_white",
    }
    return (pixels == 0).astype(np.float32), metadata


def iccad_m1_test3_target(grid_size: int, pixel_size_nm: float):
    """Rasterize the repository benchmark, then center its M1 geometry at the same pitch."""
    native_size = ceil(NATIVE_FIELD_NM / pixel_size_nm)
    native = layouts.rasterize_glp(M1_TEST3, grid_size=native_size, pixel_size_nm=pixel_size_nm)
    occupied_rows, occupied_cols = np.nonzero(native)
    if not occupied_rows.size:
        raise ValueError(f"No occupied pixels in {M1_TEST3}")

    y0, y1 = int(occupied_rows.min()), int(occupied_rows.max()) + 1
    x0, x1 = int(occupied_cols.min()), int(occupied_cols.max()) + 1
    height, width = y1 - y0, x1 - x0
    if height > grid_size or width > grid_size:
        raise ValueError(f"M1 geometry {width}x{height} does not fit the {grid_size}x{grid_size} grid")

    target_y = (grid_size - height) // 2
    target_x = (grid_size - width) // 2
    target = np.zeros((grid_size, grid_size), dtype=native.dtype)
    target[target_y:target_y + height, target_x:target_x + width] = native[y0:y1, x0:x1]
    metadata = {
        "path": str(M1_TEST3),
        "layer": "M1",
        "native_grid_size_px": native_size,
        "source_window_origin_xy_nm": [(x0 - target_x) * pixel_size_nm,
                                       (y0 - target_y) * pixel_size_nm],
        "occupied_bbox_width_height_px": [width, height],
    }
    return target, metadata


TARGET_LAYOUTS = {
    "dataset": dataset_layout_target,
    "iccad2013-m1-test3": iccad_m1_test3_target,
}
