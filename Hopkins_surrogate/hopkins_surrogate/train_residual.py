from __future__ import annotations

import argparse
import math
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpoint import build_model_from_checkpoint
from .config import SurrogateConfig, TrainConfig, load_json, save_json
from .data import PairDataset, load_cache_meta, load_splits
from .losses import residual_total_loss
from .metrics import aggregate, evaluate_sample
from .residual import CorrectedResidualImaging, ResidualCNN
from .train import _flatten_agg
from .utils import (JsonlLogger, check_disk_space, make_run_id, resolve_device,
                    set_seed)


def _lr_lambda(warmup_steps: int, total_steps: int):
    def fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return fn


def build_hybrid_from_checkpoint(path: str | Path, device,
                                 strict: bool = True
                                 ) -> tuple[CorrectedResidualImaging, dict]:
    ckpt = torch.load(Path(path), map_location="cpu",
                      weights_only=False)
    kernel_model, _ = build_model_from_checkpoint(
        ckpt["kernel_ckpt"], device, strict=strict)
    residual = ResidualCNN(
        hidden=int(ckpt["hidden"]), delta_max=float(ckpt["delta_max"]),
        downscale=int(ckpt["downscale"])).to(device)
    residual.load_state_dict(ckpt["residual_state"])
    hybrid = CorrectedResidualImaging(
        kernel_model, residual, freeze_kernel=True, detach_socs=False)
    return hybrid, ckpt


@torch.no_grad()
def validate(hybrid: CorrectedResidualImaging, loader: DataLoader,
             cfg: SurrogateConfig, s: float, pixel_nm: float,
             compute_contour: bool = True) -> dict:
    hybrid.residual.eval()
    rows_base, rows_kernel, rows_hybrid = [], [], []
    for batch in loader:
        mask = batch["mask"].to(
            next(hybrid.residual.parameters()).device).float()
        target = batch["target"].to(mask.device).float()
        binary = batch["binary"].to(mask.device).float()
        for i in range(mask.shape[0]):
            base = hybrid.kernel_model(mask[i], use_correction=False)
            kernel = hybrid.kernel_model(mask[i], use_correction=True)
            delta = hybrid.residual(mask[i], kernel / s)
            hyb = (kernel + delta).clamp_min(0.0)
            kw = dict(tau=cfg.tau_dat, pixel_size_nm=pixel_nm,
                      compute_contour=compute_contour)
            rows_base.append(evaluate_sample(base, target[i], binary[i], **kw))
            rows_kernel.append(evaluate_sample(kernel, target[i], binary[i], **kw))
            rows_hybrid.append(evaluate_sample(hyb, target[i], binary[i], **kw))
    agg_h = aggregate(rows_hybrid)
    flat = _flatten_agg(agg_h, "val_")
    flat["val_score"] = (
        flat.get("val_band_mse_mean", float("nan"))
        + 0.3 * (1.0 - flat.get("val_dice_clear_mean", float("nan")))
        + 0.1 * flat.get("val_rel_mse_mean", float("nan")))
    agg_k = _flatten_agg(aggregate(rows_kernel), "val_kernel_")
    agg_b = _flatten_agg(aggregate(rows_base), "val_base_")
    flat.update(agg_k)
    flat.update(agg_b)
    return flat


def train_residual(kernel_checkpoint: str | Path, cache_dir: str | Path,
                   run_dir: str | Path, device_name: str = "cuda:0",
                   train_cfg: TrainConfig | None = None,
                   hidden: int = 24, delta_max: float = 0.10,
                   downscale: int = 4, lr: float = 1e-3,
                   lambda_res: float = 1e-3,
                   smoke: bool = False,
                   resume_checkpoint: str | Path | None = None,
                   start_epoch: int = 0,
                   schedule_epochs: int | None = None,
                   init_best_score: float | None = None,
                   init_best_epoch: int | None = None,
                   min_delta: float = 0.0) -> dict:
    cache_dir = Path(cache_dir)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    check_disk_space(run_dir, 0.5)
    tc = train_cfg or TrainConfig()
    if smoke:
        tc = replace(tc, epochs=min(tc.epochs, 2),
                     limit_train=64, limit_val=16,
                     batch_size=min(tc.batch_size, 2))
    device = resolve_device(device_name)
    set_seed(tc.seed)

    kernel_ckpt = torch.load(Path(kernel_checkpoint), map_location="cpu",
                             weights_only=False)
    kernel_model, kernel_ckpt = build_model_from_checkpoint(
        kernel_ckpt, device)
    kernel_model.eval()
    cfg = SurrogateConfig.from_dict(kernel_ckpt["surrogate_config"])
    s = float(kernel_ckpt["calibration"]["s"])
    residual = ResidualCNN(
        hidden=int(hidden), delta_max=float(delta_max),
        downscale=int(downscale)).to(device)
    resume_ckpt = None
    if resume_checkpoint is not None:
        resume_ckpt = torch.load(Path(resume_checkpoint), map_location="cpu",
                                 weights_only=False)
        if resume_ckpt["kernel_ckpt"].get("fingerprint") != \
                kernel_ckpt.get("fingerprint"):
            raise ValueError(
                "resume checkpoint was trained on a different kernel "
                "checkpoint (fingerprint mismatch)")
        for key, value in (("hidden", hidden), ("downscale", downscale)):
            if int(resume_ckpt.get(key, value)) != int(value):
                raise ValueError(
                    f"resume checkpoint {key}={resume_ckpt.get(key)} "
                    f"!= {value}")
        if abs(float(resume_ckpt.get("delta_max", delta_max))
               - float(delta_max)) > 1e-12:
            raise ValueError(
                f"resume checkpoint delta_max={resume_ckpt.get('delta_max')} "
                f"!= {delta_max}")
        residual.load_state_dict(resume_ckpt["residual_state"])
    hybrid = CorrectedResidualImaging(
        kernel_model, residual, freeze_kernel=True, detach_socs=True)

    cache_meta = load_cache_meta(cache_dir)
    splits = load_splits(cache_dir)
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
        num_workers=tc.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(
        val_ds, batch_size=tc.batch_size, shuffle=False,
        num_workers=tc.num_workers, pin_memory=device.type == "cuda")

    optimizer = torch.optim.AdamW(
        residual.parameters(), lr=float(lr), weight_decay=1e-4)
    start_epoch = int(start_epoch)
    if start_epoch < 0:
        raise ValueError("start_epoch must be >= 0")
    steps_per_epoch = len(train_loader)
    if schedule_epochs is None:
        schedule_epochs = start_epoch + tc.epochs
    schedule_epochs = int(schedule_epochs)
    if schedule_epochs < start_epoch + tc.epochs:
        raise ValueError(
            f"schedule_epochs={schedule_epochs} is shorter than "
            f"start_epoch + epochs={start_epoch + tc.epochs}")
    total_steps = max(1, schedule_epochs * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _lr_lambda(200, total_steps))
    if start_epoch:
        for _ in range(start_epoch * steps_per_epoch):
            scheduler.step()
    pixel_nm = float(cache_meta["pixel_size_nm"])

    save_json(run_dir / "config_snapshot.json", {
        "kernel_checkpoint": str(Path(kernel_checkpoint).resolve()),
        "kernel_fingerprint": kernel_ckpt["fingerprint"],
        "surrogate_config": cfg.to_dict(),
        "train_config": tc.to_dict(),
        "hidden": hidden, "delta_max": delta_max, "downscale": downscale,
        "lr": lr, "lambda_res": lambda_res,
        "cache_meta": cache_meta,
        "split_hash": splits["split_hash"],
        "resume_checkpoint": (str(Path(resume_checkpoint).resolve())
                              if resume_checkpoint is not None else None),
        "start_epoch": start_epoch,
        "schedule_epochs": schedule_epochs,
        "min_delta": float(min_delta),
    })
    params = sum(p.numel() for p in residual.parameters())
    print(f"device={device} s={s:.6f} residual_params={params} "
          f"delta_max={delta_max} downscale={downscale}")
    print(f"train={len(train_ds)} val={len(val_ds)} "
          f"steps/epoch={len(train_loader)} total_steps={total_steps}")

    metrics_log = JsonlLogger(run_dir / "metrics.jsonl")
    steps_log = JsonlLogger(run_dir / "train_steps.jsonl")
    if init_best_score is None and resume_ckpt is not None:
        init_best_score = float(resume_ckpt.get("best_val", float("inf")))
    if init_best_epoch is None and resume_ckpt is not None:
        init_best_epoch = int(resume_ckpt.get("best_epoch",
                                              resume_ckpt.get("epoch", 0)))
    best_score = (float(init_best_score)
                  if init_best_score is not None else float("inf"))
    best_epoch = (int(init_best_epoch)
                  if init_best_epoch is not None else -1)
    global_step = start_epoch * steps_per_epoch
    stale = 0

    def save(path: Path, epoch: int, score: float) -> None:
        torch.save({
            "format_version": 1,
            "residual_state": residual.state_dict(),
            "kernel_ckpt": kernel_ckpt,
            "kernel_checkpoint": str(Path(kernel_checkpoint).resolve()),
            "hidden": int(hidden), "delta_max": float(delta_max),
            "downscale": int(downscale),
            "surrogate_config": cfg.to_dict(),
            "train_config": tc.to_dict(),
            "calibration": kernel_ckpt["calibration"],
            "split_hash": splits["split_hash"],
            "epoch": int(epoch), "best_val": float(score),
            "best_epoch": int(best_epoch),
        }, path)

    for epoch in range(tc.epochs):
        run_epoch = epoch + 1
        ge = start_epoch + run_epoch
        hybrid.residual.train()
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
                with torch.no_grad():
                    socs = hybrid.kernel_model(mask[i], use_correction=True)
                delta = hybrid.residual(mask[i], socs / s)
                pred = (socs + delta).clamp_min(0.0)
                loss, parts = residual_total_loss(
                    pred, target[i], binary[i], delta, cfg, s,
                    lambda_res=lambda_res)
                loss_sum = loss_sum + loss
                for key, value in parts.items():
                    parts_sum[key] = parts_sum.get(key, 0.0) + value
            loss = loss_sum / float(mask.shape[0])
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                residual.parameters(), tc.grad_clip))
            optimizer.step()
            scheduler.step()
            global_step += 1
            n_batches += 1
            for key, value in parts_sum.items():
                sums[key] = sums.get(key, 0.0) + value / float(mask.shape[0])
            if global_step % 20 == 0:
                steps_log.log({
                    "epoch": ge, "step": global_step,
                    "loss": float(loss.detach()),
                    "band": parts_sum.get("band", 0.0) / mask.shape[0],
                    "dice": parts_sum.get("dice", 0.0) / mask.shape[0],
                    "res": parts_sum.get("res", 0.0) / mask.shape[0],
                    "delta_abs_mean": parts_sum.get(
                        "delta_abs_mean", 0.0) / mask.shape[0],
                    "delta_abs_max": parts_sum.get(
                        "delta_abs_max", 0.0) / mask.shape[0],
                    "grad_norm": grad_norm,
                    "lr": optimizer.param_groups[0]["lr"],
                })

        epoch_seconds = time.time() - epoch_start
        train_metrics = {k: v / max(1, n_batches) for k, v in sums.items()}
        val_metrics = validate(
            hybrid, val_loader, cfg, s, pixel_nm,
            compute_contour=(ge % max(1, tc.cd_every) == 0))
        score = val_metrics["val_score"]
        improved = score < best_score - float(min_delta)
        if improved:
            best_score = score
            best_epoch = ge
            stale = 0
        else:
            stale += 1
        metrics_log.log({
            "epoch": ge, "step": global_step,
            "epoch_seconds": epoch_seconds,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **val_metrics,
            "best_val": best_score, "best_epoch": best_epoch,
            "gpu_mem_mb": (torch.cuda.max_memory_allocated(device)
                           / 1024 ** 2 if device.type == "cuda" else 0.0),
        })
        print(f"epoch {ge}/{start_epoch + tc.epochs} "
              f"lr={optimizer.param_groups[0]['lr']:.2e} "
              f"loss={train_metrics.get('loss', float('nan')):.5f} "
              f"val_dice={val_metrics.get('val_dice_clear_mean', float('nan')):.5f} "
              f"val_iou={val_metrics.get('val_iou_clear_mean', float('nan')):.5f} "
              f"val_cd={val_metrics.get('val_contour_nm_mean', float('nan')):.3f} "
              f"kernel_dice={val_metrics.get('val_kernel_dice_clear_mean', float('nan')):.5f} "
              f"res_abs={train_metrics.get('delta_abs_mean', float('nan')):.4f} "
              f"{epoch_seconds:.1f}s")
        save(run_dir / "residual_last.pt", ge, best_score)
        if improved:
            save(run_dir / "residual_best.pt", ge, best_score)
        if stale >= tc.patience:
            print(f"early stop at epoch {ge} "
                  f"(best epoch {best_epoch}, score {best_score:.6f})")
            break

    metrics_log.close()
    steps_log.close()
    summary = {
        "best_score": best_score, "best_epoch": best_epoch,
        "last_epoch": start_epoch + run_epoch if tc.epochs else start_epoch,
        "schedule_epochs": schedule_epochs,
        "resumed_from": (str(Path(resume_checkpoint).resolve())
                         if resume_checkpoint is not None else None),
        "start_epoch": start_epoch,
        "s": s, "residual_params": params,
        "kernel_checkpoint": str(Path(kernel_checkpoint).resolve()),
    }
    save_json(run_dir / "train_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a mask-conditioned bounded residual CNN on a "
                    "frozen kernel-corrected SOCS model")
    parser.add_argument("--kernel-checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path,
                        default=Path("E:/hopkins_surrogate/cache_v1"))
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--delta-max", type=float, default=0.10)
    parser.add_argument("--downscale", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-res", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--resume-checkpoint", type=Path, default=None,
                        help="residual checkpoint to continue from")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="epochs already completed in resume checkpoint")
    parser.add_argument("--schedule-epochs", type=int, default=None,
                        help="total epochs for the LR schedule "
                             "(default: start_epoch + epochs)")
    parser.add_argument("--init-best-score", type=float, default=None)
    parser.add_argument("--init-best-epoch", type=int, default=None)
    parser.add_argument("--min-delta", type=float, default=0.0,
                        help="minimum val_score improvement to reset patience")
    args = parser.parse_args()
    run_dir = args.run_dir or (
        Path("E:/hopkins_surrogate/runs") / make_run_id("residual"))
    tc = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                     seed=args.seed, limit_train=args.limit_train,
                     limit_val=args.limit_val, patience=args.patience)
    train_residual(
        args.kernel_checkpoint, args.cache, run_dir,
        device_name=args.device, train_cfg=tc, hidden=args.hidden,
        delta_max=args.delta_max, downscale=args.downscale, lr=args.lr,
        lambda_res=args.lambda_res, smoke=args.smoke,
        resume_checkpoint=args.resume_checkpoint,
        start_epoch=args.start_epoch,
        schedule_epochs=args.schedule_epochs,
        init_best_score=args.init_best_score,
        init_best_epoch=args.init_best_epoch,
        min_delta=args.min_delta)


if __name__ == "__main__":
    main()
