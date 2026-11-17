"""Small matplotlib helpers shared by the examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a > 0.5
    b = b > 0.5
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def _panel(ax, data, title, cmap, extent, vmin=None, vmax=None):
    ax.imshow(data, origin="lower", extent=extent, cmap=cmap,
              vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def save_inference(path: Path, mask, out: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask = _np(mask)
    aerial = _np(out["aerial"])
    clear = _np(out["clear"])
    printed = _np(out["printed"])
    n = mask.shape[-1]
    half = n / 2.0
    extent = (-half, half, -half, half)
    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.5))
    _panel(axes[0], mask, "input mask (1 = opaque)", "gray_r", extent, 0, 1)
    _panel(axes[1], aerial, "aerial (.dat units)", "gray", extent, 0, 1)
    _panel(axes[2], clear, "clear (aerial > threshold)", "gray", extent, 0, 1)
    _panel(axes[3], printed, "printed (aerial < threshold)", "gray_r",
           extent, 0, 1)
    fig.suptitle(
        f"HopkinsSurrogate inference · clear fraction {clear.mean():.4f} · "
        f"threshold {out['threshold']}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor(),
                pil_kwargs={"optimize": True})
    plt.close(fig)


def save_ilt(path: Path, target, initial_printed, result,
             pixel_size_nm: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = _np(target)
    initial = _np(initial_printed)
    optimized = _np(result.optimized_printed)
    mask = _np(result.optimized_mask)
    n = target.shape[-1]
    half = n * pixel_size_nm / 2.0
    extent = (-half, half, -half, half)
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.0))
    _panel(axes[0, 0], target, "target (1 = feature)", "gray_r", extent, 0, 1)
    _panel(axes[0, 1], initial,
           f"initial printed · IoU {_iou(initial, target):.4f}",
           "gray_r", extent, 0, 1)
    _panel(axes[0, 2], optimized,
           f"optimized printed · IoU {_iou(optimized, target):.4f}",
           "gray_r", extent, 0, 1)
    _panel(axes[1, 0], mask, "optimized hard mask", "gray_r", extent, 0, 1)
    steps = [row["step"] for row in result.history]
    axes[1, 1].plot(steps, [row["loss"] for row in result.history],
                    color="#2a78d6", lw=1.8)
    axes[1, 1].set_title("optimization loss", fontsize=9)
    axes[1, 1].set_xlabel("step", fontsize=8)
    axes[1, 1].grid(alpha=0.25)
    axes[1, 2].plot(steps, [row["iou"] for row in result.history],
                    color="#1baf7a", lw=1.8)
    axes[1, 2].set_title("hard-mask feature IoU", fontsize=9)
    axes[1, 2].set_xlabel("step", fontsize=8)
    axes[1, 2].set_ylim(0.6, 1.0)
    axes[1, 2].grid(alpha=0.25)
    fig.suptitle(
        f"ILT with HopkinsSurrogate · {result.steps_completed} steps · "
        f"{result.elapsed_seconds:.2f} s · "
        f"IoU {_iou(optimized, target):.4f}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor(),
                pil_kwargs={"optimize": True})
    plt.close(fig)
