import csv
import json
import math

import matplotlib
import numpy as np
from PIL import Image
from scipy import ndimage
from hopkins.resist import hard_resist
from ilt.utils.metrics import contour_mean_distance_nm

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_binary(path, features):
    Image.fromarray(np.where(features, 0, 255).astype(np.uint8)).save(path)


def _iou(a, b):
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / union if union else 1.0


def evaluate(proxy, hard_mask, target):
    aerial = proxy(hard_mask)
    printed = hard_resist(aerial, proxy.threshold).cpu().numpy().astype(bool)
    mask = hard_mask.cpu().numpy().astype(bool)
    contour = contour_mean_distance_nm(printed, target, proxy.pixel_size_nm)
    metrics = dict(
        feature_iou=_iou(printed, target),
        clear_iou=_iou(~printed, ~target),
        contour_mean_nm=contour if math.isfinite(contour) else None,
        pixel_error_fraction=float((printed != target).mean()),
        false_positive_pixels=int((printed & ~target).sum()),
        false_negative_pixels=int((target & ~printed).sum()),
        mask_components=int(ndimage.label(mask)[1]),
        mask_feature_fraction=float(mask.mean()),
    )
    return metrics, mask, printed, aerial.cpu().numpy()


def mismatch_rgb(printed, target):
    rgb = np.full((*printed.shape, 3), 255, dtype=np.uint8)
    rgb[printed & target] = (32, 32, 32)
    rgb[printed & ~target] = (221, 69, 65)
    rgb[target & ~printed] = (47, 115, 214)
    return rgb


def save_evaluation(folder, mask, printed, aerial, target):
    save_binary(folder / "mask.png", mask)
    save_binary(folder / "proxy_printed.png", printed)
    np.save(folder / "proxy_aerial.npy", aerial)
    Image.fromarray(mismatch_rgb(printed, target)).save(folder / "mismatch.png")


def save_comparison(path, target, panels):
    width = len(panels) + 1
    fig, axes = plt.subplots(3, width, figsize=(3.15 * width, 9.1), squeeze=False,
                             layout="constrained")
    for ax in axes.flat:
        ax.set_axis_off()
    axes[0, 0].imshow(target, cmap="gray_r", vmin=0, vmax=1)
    axes[0, 0].set_title("Target layout")
    axes[1, 0].text(0.5, 0.5, "Proxy print\nfrom each mask", ha="center", va="center")
    axes[2, 0].text(0.5, 0.5, "Red: extra feature\nBlue: missing feature",
                    ha="center", va="center")
    for column, (method, mask, printed, metrics) in enumerate(panels, start=1):
        axes[0, column].imshow(mask, cmap="gray_r", vmin=0, vmax=1)
        axes[0, column].set_title(f"{method}\nmask · {metrics['seconds']:.2f} s")
        axes[1, column].imshow(printed, cmap="gray_r", vmin=0, vmax=1)
        axes[1, column].set_title(f"Proxy print · IoU {metrics['feature_iou']:.4f}")
        axes[2, column].imshow(mismatch_rgb(printed, target))
    fig.savefig(path, dpi=155)
    plt.close(fig)
