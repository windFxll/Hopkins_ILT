from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset

from .config import SurrogateConfig

THRESHOLD_DAT = 0.346585
DEFAULT_DATASET_ROOT = Path("E:/diffusio4ilt_dataset")
DEFAULT_CACHE_ROOT = Path("E:/hopkins_surrogate/cache_v1")


# ----------------------------------------------------------------------
# .dat / PNG parsing
# ----------------------------------------------------------------------

def parse_slitho_dat(path: str | Path) -> tuple[np.ndarray, dict]:
    """Parse a Sentaurus RasterData2D ASCII file.

    Returns ``(arr, meta)`` where ``arr`` has shape ``(n_rows, n_cols)`` with
    row 0 at the smallest y (bottom), matching the file order.  Use
    :func:`load_target` to get the display orientation.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = text.splitlines()

    corners = []
    array_idx = None
    size_m = size_n = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("Corner_"):
            parts = s.split(":", 1)[1].split()
            if len(parts) >= 2:
                corners.append((float(parts[0]), float(parts[1])))
        elif s.startswith("Array_2D"):
            parts = s.split(":", 1)[1].split()
            if len(parts) < 2:
                raise ValueError(f"bad Array_2D line in {path}: {s!r}")
            size_m, size_n = int(parts[0]), int(parts[1])
            array_idx = i
            break
    if array_idx is None or size_m is None or size_n is None:
        raise ValueError(f"Array_2D not found in {path}")
    if not corners:
        raise ValueError(f"Corner_* records not found in {path}")

    rows: list[list[float]] = []
    for line in lines[array_idx + 1:]:
        s = line.strip()
        if not s or s.startswith("["):
            break
        rows.append([float(v) for v in s.split()])
    if len(rows) != size_n:
        raise ValueError(
            f"{path}: expected {size_n} rows, got {len(rows)}")
    if any(len(r) != size_m for r in rows):
        bad = next(i for i, r in enumerate(rows) if len(r) != size_m)
        raise ValueError(
            f"{path}: row {bad + 1} has {len(rows[bad])} values, "
            f"expected {size_m}")
    arr = np.asarray(rows, dtype=np.float32)

    xs = [c[0] * 1000.0 for c in corners]  # um -> nm
    ys = [c[1] * 1000.0 for c in corners]
    meta = {
        "n_rows": int(size_n),
        "n_cols": int(size_m),
        "x_min_nm": float(min(xs)),
        "x_max_nm": float(max(xs)),
        "y_min_nm": float(min(ys)),
        "y_max_nm": float(max(ys)),
        "path": str(path),
    }
    return arr, meta


def load_target(dat_path: str | Path, size: int = 800,
                pixel_size_nm: float | None = None,
                shift_x_nm: float = 0.0, shift_y_nm: float = 0.0) -> np.ndarray:
    """Load ``.dat`` and resample it onto the square training grid.

    Row 0 is the top of the cell, matching ``layout/png`` and
    ``aerial_image/png`` (both use the display orientation).
    """
    arr, meta = parse_slitho_dat(dat_path)
    arr = np.flipud(arr)  # row 0 = y_max (top)

    if pixel_size_nm is None:
        pixel_size_nm = 800.0 / float(size)
    extent = pixel_size_nm * float(size)
    dst_x = (-0.5 * extent + (np.arange(size) + 0.5) * pixel_size_nm
             + float(shift_x_nm))
    dst_y = (0.5 * extent - (np.arange(size) + 0.5) * pixel_size_nm
             + float(shift_y_nm))

    n_rows, n_cols = arr.shape
    dy = (meta["y_max_nm"] - meta["y_min_nm"]) / max(1, n_rows - 1)
    dx = (meta["x_max_nm"] - meta["x_min_nm"]) / max(1, n_cols - 1)
    src_row = (meta["y_max_nm"] - dst_y) / dy
    src_col = (dst_x - meta["x_min_nm"]) / dx
    coords = np.stack(np.meshgrid(src_row, src_col, indexing="ij"), axis=0)
    out = ndimage.map_coordinates(
        arr, coords, order=1, mode="nearest", prefilter=False)
    return np.asarray(out, dtype=np.float32)


def load_target_native(dat_path: str | Path) -> np.ndarray:
    """Native 320x320 target in display orientation (top row first)."""
    arr, _ = parse_slitho_dat(dat_path)
    return np.ascontiguousarray(np.flipud(arr), dtype=np.float32)


def _read_gray(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img)


def load_layout_mask(path: str | Path, size: int = 800) -> np.ndarray:
    """Layout PNG -> uint8 mask with 1 = opaque dark feature."""
    arr = _read_gray(path)
    if arr.shape != (size, size):
        raise ValueError(f"{path}: expected {size}x{size}, got {arr.shape}")
    return (arr < 128).astype(np.uint8)


def load_aerial_binary(path: str | Path, size: int = 800) -> np.ndarray:
    """Aerial PNG -> uint8 binary with 1 = clear."""
    arr = _read_gray(path)
    if arr.shape != (size, size):
        raise ValueError(f"{path}: expected {size}x{size}, got {arr.shape}")
    return (arr >= 128).astype(np.uint8)


_BASE_RE = re.compile(r"layout_(\d+)$")


def base_to_index(base: str) -> int:
    m = _BASE_RE.fullmatch(str(base))
    if m is None:
        raise ValueError(f"unexpected layout base: {base!r}")
    return int(m.group(1)) - 1


def list_pairs(dataset_root: str | Path = DEFAULT_DATASET_ROOT) -> list[dict]:
    root = Path(dataset_root)
    layout_dir = root / "layout" / "png"
    dat_dir = root / "aerial_image" / "dat"
    png_dir = root / "aerial_image" / "png"
    if not layout_dir.is_dir():
        raise FileNotFoundError(f"layout dir not found: {layout_dir}")
    layouts = sorted(
        layout_dir.glob("layout_*.png"),
        key=lambda p: int(_BASE_RE.fullmatch(p.stem).group(1)))
    pairs = []
    for lp in layouts:
        idx = base_to_index(lp.stem)
        dat = dat_dir / f"aerial_image_{idx + 1:05d}.dat"
        apng = png_dir / f"aerial_image_{idx + 1:05d}.png"
        if not dat.is_file() or not apng.is_file():
            raise FileNotFoundError(f"missing pair for {lp.stem}: {dat} / {apng}")
        pairs.append({"index": idx, "base": lp.stem,
                      "layout": lp, "dat": dat, "aerial_png": apng})
    return pairs


def read_manifest(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


# ----------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------

def _split_hash(splits: dict) -> str:
    payload = "|".join(
        f"{k}:" + ",".join(map(str, sorted(splits[k])))
        for k in ("train", "val", "test"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_group_splits(records: list[dict], all_indices,
                      train_frac: float = 0.8, val_frac: float = 0.1,
                      seed: int = 0) -> dict:
    """Group-aware 80/10/10 split by ``(config_hash, signature)``."""
    allowed = set(int(i) for i in all_indices)
    n_total = len(allowed)
    groups: dict[tuple[str, str], list[int]] = {}
    for rec in records:
        try:
            idx = base_to_index(rec["base"])
        except (KeyError, ValueError):
            continue
        if idx not in allowed:
            continue
        key = (str(rec.get("config_hash", "")),
               str(rec.get("signature", "")))
        groups.setdefault(key, []).append(idx)

    assigned = {i for idxs in groups.values() for i in idxs}
    missing = sorted(allowed - assigned)
    for idx in missing:
        groups.setdefault(("unknown", str(idx)), []).append(idx)

    target = [train_frac * n_total, val_frac * n_total,
              (1.0 - train_frac - val_frac) * n_total]
    counts = [0, 0, 0]
    split_of_group: dict[tuple[str, str], int] = {}
    rng = np.random.default_rng(seed)
    keys = list(groups.keys())
    keys.sort(key=lambda k: (-len(groups[k]), rng.random()))
    for key in keys:
        deficits = [target[j] - counts[j] for j in range(3)]
        j = int(np.argmax(deficits))
        split_of_group[key] = j
        counts[j] += len(groups[key])

    out = {"train": [], "val": [], "test": []}
    names = ("train", "val", "test")
    for key, idxs in groups.items():
        out[names[split_of_group[key]]].extend(idxs)
    for k in out:
        out[k] = sorted(out[k])
    if not out["val"] or not out["test"]:
        raise ValueError("split produced an empty val/test; increase data or "
                         "change seed")

    signature_counts = {k: {} for k in names}
    config_counts = {k: {} for k in names}
    for key, idxs in groups.items():
        name = names[split_of_group[key]]
        sig = key[1]
        cfg = key[0]
        signature_counts[name][sig] = signature_counts[name].get(sig, 0) + len(idxs)
        config_counts[name][cfg] = config_counts[name].get(cfg, 0) + len(idxs)

    return {
        "seed": int(seed),
        "n_total": int(n_total),
        "ratio": [train_frac, val_frac, 1.0 - train_frac - val_frac],
        "group_key": "config_hash|signature",
        "counts": {k: len(v) for k, v in out.items()},
        **out,
        "signature_counts": signature_counts,
        "config_hash_counts": config_counts,
        "split_hash": _split_hash(out),
    }


def assert_no_leakage(splits: dict) -> None:
    sets = {k: set(splits[k]) for k in ("train", "val", "test")}
    names = ("train", "val", "test")
    for i in range(3):
        for j in range(i + 1, 3):
            overlap = sets[names[i]] & sets[names[j]]
            if overlap:
                raise AssertionError(
                    f"{names[i]}/{names[j]} overlap: {sorted(overlap)[:10]}")
    if sum(len(s) for s in sets.values()) != len(set().union(*sets.values())):
        raise AssertionError("duplicate indices across splits")


# ----------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_cache(dataset_root: str | Path = DEFAULT_DATASET_ROOT,
                out_dir: str | Path = DEFAULT_CACHE_ROOT,
                size: int = 800, seed: int = 0, limit: int | None = None,
                config: SurrogateConfig | None = None) -> dict:
    """Build 800x800 memmap cache and split metadata.

    The arrays are written to ``out_dir`` (normally on E:).  The function is
    idempotent only when ``overwrite`` is handled by the caller; existing
    files are overwritten.
    """
    from tqdm import tqdm

    root = Path(dataset_root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs = list_pairs(root)
    if limit is not None:
        pairs = pairs[:int(limit)]
    n = len(pairs)
    if n == 0:
        raise ValueError("no pairs found")

    pixel_size_nm = 800.0 / float(size)
    if config is None:
        config = SurrogateConfig()
    config = replace(config, grid_size=int(size),
                     pixel_size_nm=float(pixel_size_nm))

    masks = np.lib.format.open_memmap(
        out / "masks.npy", mode="w+", dtype=np.uint8, shape=(n, size, size))
    targets = np.lib.format.open_memmap(
        out / "targets.npy", mode="w+", dtype=np.float16,
        shape=(n, size, size))
    binary = np.lib.format.open_memmap(
        out / "binary.npy", mode="w+", dtype=np.uint8,
        shape=(n, size, size))

    for i, pair in enumerate(tqdm(pairs, desc="cache", unit="pair")):
        mask = load_layout_mask(pair["layout"], size=size)
        target = load_target(pair["dat"], size=size,
                             pixel_size_nm=pixel_size_nm)
        b = load_aerial_binary(pair["aerial_png"], size=size)
        masks[i] = mask
        targets[i] = target.astype(np.float16)
        binary[i] = b
    masks.flush()
    targets.flush()
    binary.flush()
    del masks, targets, binary

    manifest_path = root / "layout" / "manifest.jsonl"
    if manifest_path.is_file():
        records = read_manifest(manifest_path)
        manifest_sha = _sha256_file(manifest_path)
    else:
        records = []
        manifest_sha = ""
    splits = make_group_splits(
        records, [p["index"] for p in pairs], seed=seed)
    assert_no_leakage(splits)

    cache_meta = {
        "dataset_root": str(root),
        "n": n,
        "shape": [size, size],
        "dtypes": {"masks": "uint8", "targets": "float16",
                   "binary": "uint8"},
        "mask_semantics": "1=opaque",
        "binary_semantics": "1=clear",
        "pixel_size_nm": pixel_size_nm,
        "grid_size": int(size),
        "threshold_dat": float(config.tau_dat),
        "target_resample": "bilinear_physical_coords",
        "manifest_sha256": manifest_sha,
        "split_hash": splits["split_hash"],
        "config": config.to_dict(),
    }
    (out / "splits.json").write_text(
        json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "cache_meta.json").write_text(
        json.dumps(cache_meta, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return cache_meta


def load_cache_meta(cache_dir: str | Path) -> dict:
    return json.loads((Path(cache_dir) / "cache_meta.json").read_text(
        encoding="utf-8"))


def load_splits(cache_dir: str | Path) -> dict:
    return json.loads((Path(cache_dir) / "splits.json").read_text(
        encoding="utf-8"))


# ----------------------------------------------------------------------
# Scale calibration
# ----------------------------------------------------------------------

def estimate_clear_scale(cache_dir: str | Path, split: str = "train",
                         max_images: int = 200, min_distance_nm: float = 200.0,
                         min_pixels: int = 500) -> dict:
    """Estimate the global ``aerial_scale`` from clear regions on ``split``."""
    cache_dir = Path(cache_dir)
    meta = load_cache_meta(cache_dir)
    splits = load_splits(cache_dir)
    pixel_nm = float(meta["pixel_size_nm"])
    indices = splits[split][:int(max_images)]
    masks = np.load(cache_dir / "masks.npy", mmap_mode="r")
    targets = np.load(cache_dir / "targets.npy", mmap_mode="r")

    q90, q95, q99, med = [], [], [], []
    for idx in indices:
        mask = np.asarray(masks[idx])
        target = np.asarray(targets[idx], dtype=np.float32)
        clear = mask == 0
        dist = ndimage.distance_transform_edt(clear)
        far = clear & (dist > min_distance_nm / pixel_nm)
        if int(far.sum()) < min_pixels:
            far = clear
        if int(far.sum()) == 0:
            continue
        vals = target[far]
        q90.append(float(np.quantile(vals, 0.90)))
        q95.append(float(np.quantile(vals, 0.95)))
        q99.append(float(np.quantile(vals, 0.99)))
        med.append(float(np.median(vals)))

    if not q95:
        raise ValueError("not enough clear pixels to calibrate scale")
    s = float(np.median(q95))
    return {
        "s": s,
        "method": "median_over_images_of_q95_clear_far_from_features",
        "split": split,
        "n_images": len(q95),
        "min_distance_nm": float(min_distance_nm),
        "q90_median": float(np.median(q90)),
        "q95_median": float(np.median(q95)),
        "q99_median": float(np.median(q99)),
        "median_median": float(np.median(med)),
        "split_hash": splits["split_hash"],
        "tau_dat": float(meta.get("config", {}).get(
            "tau_dat", THRESHOLD_DAT)),
    }


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------

class PairDataset(Dataset):
    """Memmap-backed paired dataset; returns raw ``.dat``-unit targets."""

    def __init__(self, cache_dir: str | Path, split: str,
                 indices: list[int] | None = None, augment: bool = False,
                 seed: int = 0):
        cache_dir = Path(cache_dir)
        self.meta = load_cache_meta(cache_dir)
        self.masks = np.load(cache_dir / "masks.npy", mmap_mode="r")
        self.targets = np.load(cache_dir / "targets.npy", mmap_mode="r")
        self.binary = np.load(cache_dir / "binary.npy", mmap_mode="r")
        if indices is None:
            indices = load_splits(cache_dir)[split]
        self.indices = [int(i) for i in indices]
        self.augment = bool(augment)
        self.rng = np.random.default_rng(int(seed))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        idx = self.indices[i]
        mask = np.array(self.masks[idx], dtype=np.float32)
        target = np.array(self.targets[idx], dtype=np.float32)
        binary = np.array(self.binary[idx], dtype=np.float32)
        if self.augment:
            k = int(self.rng.integers(0, 4))
            flip = bool(self.rng.integers(0, 2))
            if k:
                mask = np.rot90(mask, k)
                target = np.rot90(target, k)
                binary = np.rot90(binary, k)
            if flip:
                mask = np.fliplr(mask)
                target = np.fliplr(target)
                binary = np.fliplr(binary)
        return {
            "index": int(idx),
            "mask": np.ascontiguousarray(mask),
            "target": np.ascontiguousarray(target),
            "binary": np.ascontiguousarray(binary),
        }
