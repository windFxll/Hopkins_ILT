from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .config import SurrogateConfig
from .data import PairDataset, load_cache_meta, load_splits
from .evaluate import _plot_overview, _signature_map
from .metrics import aggregate, bootstrap_ci, evaluate_sample
from .train_residual import build_hybrid_from_checkpoint
from .utils import resolve_device


def evaluate(cache_dir: str | Path, residual_checkpoint: str | Path,
             split: str, out_dir: str | Path, device_name: str = "cuda:0",
             max_samples: int | None = None,
             overview_samples: int = 8) -> dict:
    cache_dir = Path(cache_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    hybrid, ckpt = build_hybrid_from_checkpoint(
        residual_checkpoint, device)
    hybrid.residual.eval()
    cfg = SurrogateConfig.from_dict(ckpt["surrogate_config"])
    s = float(ckpt["calibration"]["s"])
    use_correction = bool(ckpt.get("use_correction", True))
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
            base = hybrid.kernel_model(mask, use_correction=False)
            kernel = (hybrid.kernel_model(mask, use_correction=True)
                      if use_correction else base)
            delta = hybrid.residual(mask, kernel / s)
            hyb = (kernel + delta).clamp_min(0.0)
            kw = dict(tau=cfg.tau_dat, pixel_size_nm=pixel_nm)
            mb = evaluate_sample(base, target, binary, **kw)
            mk = evaluate_sample(kernel, target, binary, **kw)
            mh = evaluate_sample(hyb, target, binary, **kw)
            idx = int(item["index"])
            row = {
                "index": idx,
                "signature": sig_map.get(idx, ""),
                **{f"base_{k}": v for k, v in mb.items()},
                **{f"kernel_{k}": v for k, v in mk.items()},
                **{f"hybrid_{k}": v for k, v in mh.items()},
                "delta_vs_kernel_iou_clear": mh["iou_clear"] - mk["iou_clear"],
                "delta_vs_kernel_dice_clear": mh["dice_clear"] - mk["dice_clear"],
                "delta_vs_kernel_contour_nm": mh["contour_nm"] - mk["contour_nm"],
                "delta_vs_kernel_band_mse": mh["band_mse"] - mk["band_mse"],
                "delta_vs_base_iou_clear": mh["iou_clear"] - mb["iou_clear"],
                "delta_vs_base_dice_clear": mh["dice_clear"] - mb["dice_clear"],
                "delta_vs_base_contour_nm": mh["contour_nm"] - mb["contour_nm"],
                "delta_vs_base_band_mse": mh["band_mse"] - mb["band_mse"],
            }
            rows.append(row)
            if len(overview) < overview_samples:
                overview.append((idx, item,
                                 kernel.detach().cpu().numpy(),
                                 hyb.detach().cpu().numpy()))

    keys = ["iou_clear", "dice_clear", "iou_feature", "dice_feature",
            "band_mse", "rel_mse", "contour_nm"]
    base_agg = aggregate([{k: r[f"base_{k}"] for k in keys} for r in rows])
    kernel_agg = aggregate([{k: r[f"kernel_{k}"] for k in keys} for r in rows])
    hybrid_agg = aggregate([{k: r[f"hybrid_{k}"] for k in keys} for r in rows])
    dk = {k: [r[f"delta_vs_kernel_{k}"] for r in rows] for k in
          ("iou_clear", "dice_clear", "contour_nm", "band_mse")}
    db = {k: [r[f"delta_vs_base_{k}"] for r in rows] for k in
          ("iou_clear", "dice_clear", "contour_nm", "band_mse")}
    summary = {
        "split": split, "n": len(rows), "s": s,
        "tau_dat": cfg.tau_dat,
        "checkpoint": str(residual_checkpoint),
        "baseline": base_agg, "kernel_only": kernel_agg,
        "hybrid": hybrid_agg,
        "delta_vs_kernel": {k: aggregate([{"v": v} for v in vals])["v"]
                            for k, vals in dk.items()},
        "delta_vs_kernel_ci95": {k: bootstrap_ci(vals)
                                 for k, vals in dk.items()},
        "delta_vs_base": {k: aggregate([{"v": v} for v in vals])["v"]
                          for k, vals in db.items()},
        "delta_vs_base_ci95": {k: bootstrap_ci(vals)
                               for k, vals in db.items()},
        "fraction_improved_vs_kernel": float(np.mean(
            [r["delta_vs_kernel_iou_clear"] > 0 for r in rows])),
        "fraction_bad_vs_kernel": float(np.mean(
            [r["delta_vs_kernel_iou_clear"] < -0.02 for r in rows])),
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
    return summary


def _write_report(path: Path, summary: dict) -> None:
    base = summary["baseline"]
    kernel = summary["kernel_only"]
    hyb = summary["hybrid"]
    dk = summary["delta_vs_kernel"]
    db = summary["delta_vs_base"]
    ck = summary["delta_vs_kernel_ci95"]
    lines = [
        f"# Residual CNN evaluation ({summary['split']})",
        "",
        f"- samples: {summary['n']}",
        f"- s: {summary['s']:.6f}",
        f"- threshold: {summary['tau_dat']}",
        f"- checkpoint: `{summary['checkpoint']}`",
        "",
        "| metric | baseline SOCS | kernel-only | kernel+CNN | Δ vs kernel | Δ vs baseline |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in ["iou_clear", "dice_clear", "iou_feature", "dice_feature",
                "band_mse", "rel_mse", "contour_nm"]:
        b = base.get(key, {}).get("mean", float("nan"))
        k = kernel.get(key, {}).get("mean", float("nan"))
        h = hyb.get(key, {}).get("mean", float("nan"))
        dvk = dk.get(key, {}).get("mean", float("nan"))
        dvb = db.get(key, {}).get("mean", float("nan"))
        if not np.isfinite(dvk):
            dvk = h - k
        if not np.isfinite(dvb):
            dvb = h - b
        lines.append(f"| {key} | {b:.6f} | {k:.6f} | {h:.6f} | "
                     f"{dvk:+.6f} | {dvb:+.6f} |")
    lines += [
        "",
        f"- fraction improved vs kernel (clear IoU): "
        f"{summary['fraction_improved_vs_kernel']:.3f}",
        f"- fraction worse vs kernel by >0.02 IoU: "
        f"{summary['fraction_bad_vs_kernel']:.3f}",
    ]
    for key in ("iou_clear", "dice_clear", "contour_nm"):
        ci = ck[key]
        lines.append(f"- Δ vs kernel {key}: "
                     f"{ci['mean']:+.6f} [{ci['lo']:+.6f}, {ci['hi']:+.6f}]")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plots(out_dir: Path, rows: list[dict], overview: list,
           tau: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].hist([r["delta_vs_kernel_iou_clear"] for r in rows], bins=40)
    axes[0].set_title("hybrid - kernel-only: clear IoU")
    x = [r["kernel_iou_clear"] for r in rows]
    y = [r["hybrid_iou_clear"] for r in rows]
    axes[1].scatter(x, y, s=4, alpha=0.5)
    lim = [min(x + y), max(x + y)]
    axes[1].plot(lim, lim, color="red", linewidth=0.8)
    axes[1].set_xlabel("kernel-only IoU")
    axes[1].set_ylabel("kernel+CNN IoU")
    axes[1].set_title("per-sample improvement")
    fig.tight_layout()
    fig.savefig(out_dir / "hybrid_delta.png", dpi=140)
    plt.close(fig)

    for idx, item, kernel, hyb in overview:
        _plot_overview(
            out_dir / f"overview_{idx:05d}.png",
            item["mask"], item["target"], kernel, hyb,
            item["binary"], tau)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate kernel-only vs kernel+residual CNN")
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
        "hybrid_vs_kernel_delta_iou": summary["delta_vs_kernel"]["iou_clear"],
        "hybrid_vs_kernel_delta_dice": summary["delta_vs_kernel"]["dice_clear"],
        "hybrid_vs_kernel_delta_cd": summary["delta_vs_kernel"]["contour_nm"],
    }, indent=2))


if __name__ == "__main__":
    main()
