from __future__ import annotations

import argparse
import math
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpoint import collect_rng_states, save_checkpoint
from .config import SurrogateConfig, TrainConfig, load_json, save_json
from .data import (PairDataset, estimate_clear_scale, load_cache_meta,
                   load_splits)
from .imaging import CorrectedSocsImaging
from .losses import total_loss
from .metrics import aggregate, evaluate_sample
from .utils import (JsonlLogger, check_disk_space, git_commit, make_run_id,
                    resolve_device, set_seed)


def _flatten_agg(agg: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in agg.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out[f"{name}_mean"] = value.get("mean")
            out[f"{name}_std"] = value.get("std")
            out[f"{name}_median"] = value.get("median")
            out[f"{name}_count"] = value.get("count")
        else:
            out[name] = value
    return out


def _lr_lambda(warmup_steps: int, total_steps: int):
    def fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return fn


def build_model(cache_meta: dict, calibration: dict, device,
                num_orders: int | None = None,
                basis_size: int | None = None) -> tuple[
                    CorrectedSocsImaging, SurrogateConfig, float]:
    cfg = SurrogateConfig.from_dict(cache_meta.get("config", {}))
    s = float(calibration["s"])
    cfg = replace(cfg, aerial_scale=s)
    if num_orders is not None:
        cfg = replace(cfg, num_orders=int(num_orders))
    if basis_size is not None:
        cfg = replace(cfg, basis_size=int(basis_size))
    optics = cfg.optics(s)
    model = CorrectedSocsImaging(
        optics, device,
        num_orders=cfg.num_orders,
        basis_size=cfg.basis_size,
        alpha_max=cfg.alpha_max,
        alpha_init=cfg.alpha_init,
        c_init_std=cfg.c_init_std,
        window_inner=cfg.window_inner,
        window_outer=cfg.window_outer,
        seed=None,
    )
    return model, cfg, s


@torch.no_grad()
def validate(model: CorrectedSocsImaging, loader: DataLoader,
             cfg: SurrogateConfig, s: float, pixel_nm: float,
             compute_contour: bool = True) -> dict:
    model.eval()
    rows = []
    for batch in loader:
        mask = batch["mask"].to(next(model.parameters()).device).float()
        target = batch["target"].to(mask.device).float()
        binary = batch["binary"].to(mask.device).float()
        for i in range(mask.shape[0]):
            pred = model(mask[i], use_correction=True)
            rows.append(evaluate_sample(
                pred, target[i], binary[i], tau=cfg.tau_dat,
                pixel_size_nm=pixel_nm, compute_contour=compute_contour))
    agg = aggregate(rows)
    flat = _flatten_agg(agg, "val_")
    flat["val_score"] = (
        flat.get("val_band_mse_mean", float("nan"))
        + 0.3 * (1.0 - flat.get("val_dice_clear_mean", float("nan")))
        + 0.1 * flat.get("val_rel_mse_mean", float("nan")))
    if not compute_contour:
        flat.pop("val_contour_nm_mean", None)
        flat.pop("val_contour_nm_std", None)
        flat.pop("val_contour_nm_median", None)
        flat.pop("val_contour_nm_count", None)
    return flat


def train(cache_dir: str | Path, calibration_path: str | Path,
          run_dir: str | Path, device_name: str = "cuda:0",
          train_cfg: TrainConfig | None = None,
          num_orders: int | None = None,
          lr_c: float | None = None, lr_alpha: float | None = None,
          lambda_bin: float | None = None,
          lambda_rel: float | None = None,
          lambda_smooth: float | None = None,
          rel_norm_max: float | None = None,
          rel_norm_mean_max: float | None = None,
          rel_norm_target: float | None = None,
          basis_size: int | None = None,
          smoke: bool = False) -> dict:
    cache_dir = Path(cache_dir)
    run_dir = Path(run_dir)
    calib = load_json(calibration_path)
    cache_meta = load_cache_meta(cache_dir)
    splits = load_splits(cache_dir)
    tc = train_cfg or TrainConfig()
    if smoke:
        tc = replace(tc, epochs=min(tc.epochs, 2),
                     limit_train=64, limit_val=16, batch_size=min(tc.batch_size, 4))
    if lr_c is not None:
        tc = replace(tc, lr_c=float(lr_c))
    if lr_alpha is not None:
        tc = replace(tc, lr_alpha=float(lr_alpha))

    device = resolve_device(device_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    check_disk_space(run_dir, 0.5)
    set_seed(tc.seed)

    model, cfg, s = build_model(cache_meta, calib, device,
                                num_orders=num_orders,
                                basis_size=basis_size)
    if lambda_bin is not None:
        cfg = replace(cfg, lambda_bin=float(lambda_bin))
    if lambda_rel is not None:
        cfg = replace(cfg, lambda_rel=float(lambda_rel))
    if lambda_smooth is not None:
        cfg = replace(cfg, lambda_smooth=float(lambda_smooth))
    if rel_norm_max is not None:
        cfg = replace(cfg, rel_norm_max=float(rel_norm_max))
    if rel_norm_mean_max is not None:
        cfg = replace(cfg, rel_norm_mean_max=float(rel_norm_mean_max))
    if rel_norm_target is not None:
        cfg = replace(cfg, rel_norm_target=float(rel_norm_target))

    train_idx = splits["train"]
    val_idx = splits["val"]
    if tc.limit_train is not None:
        train_idx = train_idx[:int(tc.limit_train)]
    if tc.limit_val is not None:
        val_idx = val_idx[:int(tc.limit_val)]
    train_ds = PairDataset(cache_dir, "train", indices=train_idx,
                           augment=True, seed=tc.seed)
    val_ds = PairDataset(cache_dir, "val", indices=val_idx,
                         augment=False, seed=tc.seed)
    train_loader = DataLoader(
        train_ds, batch_size=tc.batch_size, shuffle=True,
        num_workers=tc.num_workers, pin_memory=device.type == "cuda",
        drop_last=False)
    val_loader = DataLoader(
        val_ds, batch_size=tc.batch_size, shuffle=False,
        num_workers=tc.num_workers, pin_memory=device.type == "cuda")

    corr = model.correction
    optimizer = torch.optim.AdamW([
        {"params": [corr.c_real, corr.c_imag], "lr": tc.lr_c,
         "weight_decay": tc.weight_decay_c},
        {"params": [corr.alpha_logit], "lr": tc.lr_alpha,
         "weight_decay": 0.0},
    ])
    total_steps = max(1, tc.epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _lr_lambda(tc.warmup_steps, total_steps))

    pixel_nm = float(cache_meta["pixel_size_nm"])
    save_json(run_dir / "config_snapshot.json", {
        "surrogate_config": cfg.to_dict(),
        "train_config": tc.to_dict(),
        "cache_meta": cache_meta,
        "calibration": calib,
        "split_hash": splits["split_hash"],
        "git_commit": git_commit(),
    })

    params = sum(p.numel() for p in model.parameters())
    print(f"device={device} s={s:.6f} K={model.num_orders} "
          f"rank={model.rank} subgrid={model.subgrid_shape} "
          f"grid={cache_meta['grid_size']} params={params}")
    print(f"train={len(train_ds)} val={len(val_ds)} "
          f"steps/epoch={len(train_loader)} total_steps={total_steps}")

    metrics_log = JsonlLogger(run_dir / "metrics.jsonl")
    steps_log = JsonlLogger(run_dir / "train_steps.jsonl")
    best_score = float("inf")
    best_epoch = -1
    global_step = 0
    stale = 0

    for epoch in range(tc.epochs):
        model.train()
        corr.set_enabled(True)
        epoch_start = time.time()
        sums = {}
        n_batches = 0
        for batch in train_loader:
            mask = batch["mask"].to(device).float()
            target = batch["target"].to(device).float()
            binary = batch["binary"].to(device).float()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            parts_sum = {}
            for i in range(mask.shape[0]):
                pred = model(mask[i], use_correction=True)
                loss, parts = total_loss(
                    pred, target[i], binary[i], corr, cfg, s,
                    base_kernels=model.base.kernels)
                loss_sum = loss_sum + loss
                for key, value in parts.items():
                    parts_sum[key] = parts_sum.get(key, 0.0) + value
            loss = loss_sum / float(mask.shape[0])
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                [corr.c_real, corr.c_imag, corr.alpha_logit], tc.grad_clip))
            optimizer.step()
            corr.project_relative_norm(
                model.base.kernels,
                mean_max=cfg.rel_norm_mean_max,
                mode_max=cfg.rel_norm_max)
            scheduler.step()
            global_step += 1
            n_batches += 1
            for key, value in parts_sum.items():
                sums[key] = sums.get(key, 0.0) + value / float(mask.shape[0])
            if global_step % 20 == 0:
                rel = corr.relative_norm(model.base.kernels)
                steps_log.log({
                    "epoch": epoch + 1,
                    "step": global_step,
                    "loss": float(loss.detach()),
                    "band": parts_sum.get("band", 0.0) / mask.shape[0],
                    "dice": parts_sum.get("dice", 0.0) / mask.shape[0],
                    "rel": parts_sum.get("rel", 0.0) / mask.shape[0],
                    "smooth": parts_sum.get("smooth", 0.0) / mask.shape[0],
                    "alpha": float(corr.alpha.detach()),
                    "rel_norm_mean": float(rel.mean()),
                    "rel_norm_max": float(rel.max()),
                    "grad_norm": grad_norm,
                    "lr_c": optimizer.param_groups[0]["lr"],
                })

        epoch_seconds = time.time() - epoch_start
        train_metrics = {k: v / max(1, n_batches) for k, v in sums.items()}
        rel = corr.relative_norm(model.base.kernels).detach()
        val_metrics = validate(
            model, val_loader, cfg, s, pixel_nm,
            compute_contour=((epoch + 1) % max(1, tc.cd_every) == 0))
        score = val_metrics["val_score"]
        improved = score < best_score
        if improved:
            best_score = score
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1

        payload = {
            "epoch": epoch + 1,
            "step": global_step,
            "epoch_seconds": epoch_seconds,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **val_metrics,
            "alpha": float(corr.alpha.detach()),
            "rel_norm_mean": float(rel.mean()),
            "rel_norm_max": float(rel.max()),
            "best_val": best_score,
            "best_epoch": best_epoch,
            "lr_c": optimizer.param_groups[0]["lr"],
            "gpu_mem_mb": (torch.cuda.max_memory_allocated(device)
                           / 1024 ** 2 if device.type == "cuda" else 0.0),
        }
        metrics_log.log(payload)
        print(f"epoch {epoch + 1}/{tc.epochs} "
              f"loss={train_metrics.get('loss', float('nan')):.5f} "
              f"val_dice={val_metrics.get('val_dice_clear_mean', float('nan')):.5f} "
              f"val_iou={val_metrics.get('val_iou_clear_mean', float('nan')):.5f} "
              f"val_cd={val_metrics.get('val_contour_nm_mean', float('nan')):.3f} "
              f"alpha={payload['alpha']:.5f} "
              f"rel={payload['rel_norm_mean']:.4f} "
              f"{epoch_seconds:.1f}s")

        save_checkpoint(run_dir / "checkpoint_last.pt", model, optimizer,
                        scheduler, epoch + 1, global_step, best_score,
                        tc, cfg, calib, splits["split_hash"],
                        collect_rng_states())
        if improved:
            save_checkpoint(run_dir / "checkpoint_best.pt", model, optimizer,
                            scheduler, epoch + 1, global_step, best_score,
                            tc, cfg, calib, splits["split_hash"],
                            collect_rng_states())
        if stale >= tc.patience:
            print(f"early stop at epoch {epoch + 1} "
                  f"(best epoch {best_epoch}, score {best_score:.6f})")
            break

    metrics_log.close()
    steps_log.close()
    summary = {
        "best_score": best_score,
        "best_epoch": best_epoch,
        "s": s,
        "num_orders": model.num_orders,
        "rank": model.rank,
        "params": params,
        "run_dir": str(run_dir),
    }
    save_json(run_dir / "train_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a lightweight SOCS frequency-domain correction")
    parser.add_argument("--cache", type=Path,
                        default=Path("E:/hopkins_surrogate/cache_v1"))
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-orders", type=int, default=16)
    parser.add_argument("--basis-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr-c", type=float, default=None)
    parser.add_argument("--lr-alpha", type=float, default=None)
    parser.add_argument("--lambda-bin", type=float, default=None)
    parser.add_argument("--lambda-rel", type=float, default=None)
    parser.add_argument("--lambda-smooth", type=float, default=None)
    parser.add_argument("--rel-norm-max", type=float, default=None)
    parser.add_argument("--rel-norm-mean-max", type=float, default=None)
    parser.add_argument("--rel-norm-target", type=float, default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    calibration = args.calibration or (args.cache / "calibration.json")
    if not Path(calibration).is_file():
        raise FileNotFoundError(
            f"calibration not found: {calibration}; run calibrate first")
    run_dir = args.run_dir or (
        Path("E:/hopkins_surrogate/runs") / make_run_id("exp"))
    tc = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                     seed=args.seed, limit_train=args.limit_train,
                     limit_val=args.limit_val)
    train(args.cache, calibration, run_dir, device_name=args.device,
          train_cfg=tc, num_orders=args.num_orders,
          lr_c=args.lr_c, lr_alpha=args.lr_alpha,
          lambda_bin=args.lambda_bin, lambda_rel=args.lambda_rel,
          lambda_smooth=args.lambda_smooth,
          rel_norm_max=args.rel_norm_max,
          rel_norm_mean_max=args.rel_norm_mean_max,
          rel_norm_target=args.rel_norm_target,
          basis_size=args.basis_size,
          smoke=args.smoke)


if __name__ == "__main__":
    main()
