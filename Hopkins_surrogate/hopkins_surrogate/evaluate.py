from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoint import build_model_from_checkpoint
from .config import SurrogateConfig
from .data import (PairDataset, load_cache_meta, load_splits, read_manifest,
                   base_to_index)
from .metrics import aggregate, bootstrap_ci, evaluate_sample
from .utils import resolve_device


def _signature_map(cache_meta: dict) -> dict[int, str]:
    root = Path(cache_meta.get("dataset_root", ""))
    manifest = root / "layout" / "manifest.jsonl"
    if not manifest.is_file():
        return {}
    out = {}
    for rec in read_manifest(manifest):
        try:
            out[base_to_index(rec["base"])] = str(rec.get("signature", ""))
        except (KeyError, ValueError):
            continue
    return out


def _mean(agg: dict, key: str) -> float:
    return float(agg.get(key, {}).get("mean", float("nan")))


def _plot_overview(path: Path, mask: np.ndarray, target: np.ndarray,
                   pred_base: np.ndarray, pred_corr: np.ndarray,
                   binary: np.ndarray, tau: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    panels = [
        (mask, "layout mask (black=opaque)", "gray_r", None),
        (target, "target (.dat)", "inferno", None),
        (pred_base, "baseline SOCS", "inferno", None),
        (pred_corr, "corrected", "inferno", None),
        (pred_corr - target, "corrected - target", "coolwarm",
         (-0.15, 0.15)),
        ((pred_corr > tau) ^ (binary > 0.5),
         "binary mismatch (red)", "Reds", (0, 1)),
    ]
    for ax, (data, title, cmap, limits) in zip(axes.ravel(), panels):
        im = ax.imshow(data, cmap=cmap)
        if limits is not None:
            im.set_clim(*limits)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_training(run_dir: Path, out_dir: Path) -> None:
    src = run_dir / "metrics.jsonl"
    if not src.is_file():
        return
    rows = [json.loads(line) for line in
            src.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [r.get("epoch") for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    panels = [
        ("train_loss", "train loss"),
        ("val_dice_clear_mean", "val clear Dice"),
        ("val_contour_nm_mean", "val contour (nm)"),
        ("rel_norm_mean", "relative correction norm"),
    ]
    for ax, (key, title) in zip(axes.ravel(), panels):
        vals = [r.get(key, np.nan) for r in rows]
        ax.plot(epochs, vals, marker="o", ms=2)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=120)
    plt.close(fig)


def evaluate(cache_dir: str | Path, checkpoint_path: str | Path,
             split: str, out_dir: str | Path, device_name: str = "cuda:0",
             max_samples: int | None = None,
             overview_samples: int = 8) -> dict:
    cache_dir = Path(cache_dir)
    checkpoint_path = Path(checkpoint_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)

    ckpt = torch.load(checkpoint_path, map_location="cpu",
                      weights_only=False)
    model, ckpt = build_model_from_checkpoint(ckpt, device)
    model.eval()
    cfg = SurrogateConfig.from_dict(ckpt["surrogate_config"])
    s = float(ckpt["calibration"]["s"])
    cache_meta = load_cache_meta(cache_dir)
    splits = load_splits(cache_dir)
    pixel_nm = float(cache_meta["pixel_size_nm"])
    sig_map = _signature_map(cache_meta)

    indices = splits[split]
    if max_samples is not None:
        indices = indices[:int(max_samples)]
    ds = PairDataset(cache_dir, split, indices=indices, augment=False)

    rows = []
    overview = []
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            mask = torch.as_tensor(item["mask"], device=device).float()
            target = torch.as_tensor(item["target"], device=device).float()
            binary = torch.as_tensor(item["binary"], device=device).float()
            pred_base = model(mask, use_correction=False)
            pred_corr = model(mask, use_correction=True)
            mb = evaluate_sample(pred_base, target, binary,
                                 tau=cfg.tau_dat, pixel_size_nm=pixel_nm)
            mc = evaluate_sample(pred_corr, target, binary,
                                 tau=cfg.tau_dat, pixel_size_nm=pixel_nm)
            idx = int(item["index"])
            row = {
                "index": idx,
                "signature": sig_map.get(idx, ""),
                **{f"base_{k}": v for k, v in mb.items()},
                **{f"corr_{k}": v for k, v in mc.items()},
                "delta_iou_clear": mc["iou_clear"] - mb["iou_clear"],
                "delta_dice_clear": mc["dice_clear"] - mb["dice_clear"],
                "delta_iou_feature": mc["iou_feature"] - mb["iou_feature"],
                "delta_dice_feature": mc["dice_feature"] - mb["dice_feature"],
                "delta_band_mse": mc["band_mse"] - mb["band_mse"],
                "delta_rel_mse": mc["rel_mse"] - mb["rel_mse"],
                "delta_contour_nm": mc["contour_nm"] - mb["contour_nm"],
            }
            rows.append(row)
            if len(overview) < overview_samples:
                overview.append((
                    idx, item,
                    pred_base.detach().cpu().numpy(),
                    pred_corr.detach().cpu().numpy(),
                    mb, mc))

    keys = ["iou_clear", "dice_clear", "iou_feature", "dice_feature",
            "band_mse", "rel_mse", "contour_nm"]
    base_agg = aggregate([{k: r[f"base_{k}"] for k in keys} for r in rows])
    corr_agg = aggregate([{k: r[f"corr_{k}"] for k in keys} for r in rows])
    delta_agg = aggregate([{k: r[f"delta_{k}"] for k in keys} for r in rows])
    ci = {k: bootstrap_ci([r[f"delta_{k}"] for r in rows])
          for k in keys}

    summary = {
        "split": split,
        "n": len(rows),
        "s": s,
        "tau_dat": cfg.tau_dat,
        "checkpoint": str(checkpoint_path),
        "baseline": base_agg,
        "corrected": corr_agg,
        "delta": delta_agg,
        "delta_ci95": ci,
        "fraction_improved_iou": float(np.mean(
            [r["delta_iou_clear"] > 0 for r in rows])),
        "fraction_bad_iou": float(np.mean(
            [r["delta_iou_clear"] < -0.02 for r in rows])),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out_dir / "per_sample.csv").open(
            "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    _write_report(out_dir / "report.md", summary)
    _plots(out_dir, rows, overview, cfg.tau_dat)
    _plot_training(checkpoint_path.parent, out_dir)
    return summary


def _write_report(path: Path, summary: dict) -> None:
    base = summary["baseline"]
    corr = summary["corrected"]
    delta = summary["delta"]
    ci = summary["delta_ci95"]
    lines = [
        f"# Evaluation report ({summary['split']})",
        "",
        f"- samples: {summary['n']}",
        f"- s (aerial_scale): {summary['s']:.6f}",
        f"- threshold (raw .dat): {summary['tau_dat']}",
        f"- checkpoint: `{summary['checkpoint']}`",
        "",
        "| metric | baseline | corrected | delta | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in ["iou_clear", "dice_clear", "iou_feature", "dice_feature",
                "band_mse", "rel_mse", "contour_nm"]:
        b = base.get(key, {}).get("mean", float("nan"))
        c = corr.get(key, {}).get("mean", float("nan"))
        d = delta.get(key, {}).get("mean", float("nan"))
        lo = ci.get(key, {}).get("lo", float("nan"))
        hi = ci.get(key, {}).get("hi", float("nan"))
        lines.append(f"| {key} | {b:.6f} | {c:.6f} | {d:+.6f} | "
                     f"[{lo:+.6f}, {hi:+.6f}] |")
    lines += [
        "",
        f"- fraction improved (clear IoU): "
        f"{summary['fraction_improved_iou']:.3f}",
        f"- fraction worse by >0.02 IoU: "
        f"{summary['fraction_bad_iou']:.3f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plots(out_dir: Path, rows: list[dict], overview: list,
           tau: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist([r["delta_iou_clear"] for r in rows], bins=40)
    axes[0].set_title("delta clear IoU")
    axes[1].hist([r["delta_contour_nm"] for r in rows
                  if np.isfinite(r["delta_contour_nm"])], bins=40)
    axes[1].set_title("delta contour (nm)")
    fig.tight_layout()
    fig.savefig(out_dir / "delta_hist.png", dpi=120)
    plt.close(fig)

    for idx, item, pred_base, pred_corr, _, _ in overview:
        _plot_overview(
            out_dir / f"overview_{idx:05d}.png",
            item["mask"], item["target"], pred_base, pred_corr,
            item["binary"], tau)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline SOCS vs corrected SOCS")
    parser.add_argument("--cache", type=Path,
                        default=Path("E:/hopkins_surrogate/cache_v1"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"),
                        default="test")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    summary = evaluate(args.cache, args.checkpoint, args.split, args.out,
                       device_name=args.device,
                       max_samples=args.max_samples)
    print(json.dumps({
        "n": summary["n"],
        "delta_iou_clear": summary["delta"]["iou_clear"],
        "delta_dice_clear": summary["delta"]["dice_clear"],
        "delta_contour_nm": summary["delta"]["contour_nm"],
    }, indent=2))


if __name__ == "__main__":
    main()
