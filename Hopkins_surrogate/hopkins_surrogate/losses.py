from __future__ import annotations

import torch

from .config import SurrogateConfig


def contour_weight(target: torch.Tensor, tau: float,
                   width: float) -> torch.Tensor:
    """Weight 1 near the threshold band and 0.1 elsewhere."""
    return 0.1 + 0.9 * torch.exp(-(((target - tau) / width) ** 2))


def band_loss(pred: torch.Tensor, target: torch.Tensor, tau: float,
              width: float) -> torch.Tensor:
    w = contour_weight(target, tau, width)
    return (w * (pred - target).square()).mean() / w.mean().clamp_min(1e-8)


def dice_loss(pred: torch.Tensor, target_binary: torch.Tensor, tau: float,
              temperature: float, eps: float = 1.0) -> torch.Tensor:
    """Soft-Dice on the clear region; ``target_binary`` is 1=clear."""
    p = torch.sigmoid((pred - tau) / temperature)
    inter = (p * target_binary).sum()
    return 1.0 - (2.0 * inter + eps) / (
        p.sum() + target_binary.sum() + eps)


def relative_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).square().sum() / (
        target.square().sum() + 1e-8)


def total_loss(pred: torch.Tensor, target: torch.Tensor,
               target_binary: torch.Tensor,
               correction, cfg: SurrogateConfig,
               scale: float,
               base_kernels: torch.Tensor | None = None
               ) -> tuple[torch.Tensor, dict]:
    """Return total loss and detached component values.

    ``pred`` and ``target`` are in raw ``.dat`` units; ``pred`` already
    includes ``aerial_scale=scale``.  ``l_smooth`` combines the relative
    kernel-correction norm with a coefficient smoothness penalty.
    """
    width = float(cfg.band_width_norm) * float(scale)
    temperature = float(cfg.dice_temp_norm) * float(scale)
    l_band = band_loss(pred, target, cfg.tau_dat, width)
    l_dice = dice_loss(pred, target_binary, cfg.tau_dat, temperature)
    l_rel = relative_loss(pred, target)
    l_coeff = correction.coefficient_smoothness()
    if base_kernels is None:
        l_kernel = torch.zeros((), device=pred.device)
    else:
        rel = correction.relative_norm(base_kernels)
        target = float(getattr(cfg, "rel_norm_target", 0.05))
        l_kernel = (rel / target).square().mean()
    l_smooth = l_kernel + 0.01 * l_coeff
    total = (l_band
             + float(cfg.lambda_bin) * l_dice
             + float(cfg.lambda_rel) * l_rel
             + float(cfg.lambda_smooth) * l_smooth)
    parts = {
        "loss": float(total.detach()),
        "band": float(l_band.detach()),
        "dice": float(l_dice.detach()),
        "rel": float(l_rel.detach()),
        "smooth": float(l_smooth.detach()),
        "kernel_rel": float(l_kernel.detach()),
        "coeff_smooth": float(l_coeff.detach()),
        "alpha": float(correction.alpha.detach()),
    }
    return total, parts


def residual_total_loss(pred: torch.Tensor, target: torch.Tensor,
                        target_binary: torch.Tensor,
                        delta: torch.Tensor, cfg: SurrogateConfig,
                        scale: float,
                        lambda_res: float = 1e-3
                        ) -> tuple[torch.Tensor, dict]:
    """Hybrid loss for the frozen kernel model + residual CNN."""
    width = float(cfg.band_width_norm) * float(scale)
    temperature = float(cfg.dice_temp_norm) * float(scale)
    l_band = band_loss(pred, target, cfg.tau_dat, width)
    l_dice = dice_loss(pred, target_binary, cfg.tau_dat, temperature)
    l_res = delta.square().mean()
    total = (l_band + float(cfg.lambda_bin) * l_dice
             + float(lambda_res) * l_res)
    parts = {
        "loss": float(total.detach()),
        "band": float(l_band.detach()),
        "dice": float(l_dice.detach()),
        "res": float(l_res.detach()),
        "delta_abs_mean": float(delta.abs().mean().detach()),
        "delta_abs_max": float(delta.abs().max().detach()),
    }
    return total, parts
