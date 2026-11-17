from __future__ import annotations

import numpy as np
from scipy import ndimage

THRESHOLD_DAT = 0.346585


def _as_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def binary_dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / denom)


def contour_mean_distance_nm(pred_bin: np.ndarray, target_bin: np.ndarray,
                             pixel_size_nm: float = 1.0) -> float:
    pred_bin = pred_bin.astype(bool)
    target_bin = target_bin.astype(bool)
    pred_edge = np.logical_xor(pred_bin, ndimage.binary_erosion(pred_bin))
    target_edge = np.logical_xor(
        target_bin, ndimage.binary_erosion(target_bin))
    if not pred_edge.any() or not target_edge.any():
        return float("nan")
    d_to_target = ndimage.distance_transform_edt(~target_edge)
    d_to_pred = ndimage.distance_transform_edt(~pred_edge)
    return float(0.5 * (d_to_target[pred_edge].mean()
                        + d_to_pred[target_edge].mean()) * pixel_size_nm)


def evaluate_sample(pred, target, target_binary=None,
                    tau: float = THRESHOLD_DAT,
                    pixel_size_nm: float = 1.0,
                    compute_contour: bool = True) -> dict:
    """Metrics for one sample; binary semantics are 1=clear."""
    pred = _as_numpy(pred).astype(np.float32)
    target = _as_numpy(target).astype(np.float32)
    pred_bin = pred > tau
    if target_binary is None:
        target_bin = target > tau
    else:
        target_bin = _as_numpy(target_binary) > 0.5

    out = {
        "iou_clear": binary_iou(pred_bin, target_bin),
        "dice_clear": binary_dice(pred_bin, target_bin),
        "iou_feature": binary_iou(~pred_bin, ~target_bin),
        "dice_feature": binary_dice(~pred_bin, ~target_bin),
        "band_mse": float(np.mean((pred - target) ** 2)),
        "rel_mse": float(np.sum((pred - target) ** 2)
                         / (np.sum(target ** 2) + 1e-8)),
    }
    if compute_contour:
        out["contour_nm"] = contour_mean_distance_nm(
            pred_bin, target_bin, pixel_size_nm)
    return out


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    out = {"n": len(rows)}
    for key in keys:
        vals = np.asarray([r[key] for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            out[key] = {"mean": float("nan"), "std": float("nan"),
                        "median": float("nan"), "count": 0}
            continue
        out[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "median": float(np.median(vals)),
            "count": int(vals.size),
        }
    return out


def bootstrap_ci(values, n_boot: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> dict:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"mean": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        means[i] = rng.choice(vals, size=vals.size, replace=True).mean()
    return {
        "mean": float(vals.mean()),
        "lo": float(np.quantile(means, alpha / 2.0)),
        "hi": float(np.quantile(means, 1.0 - alpha / 2.0)),
        "n": int(vals.size),
    }
