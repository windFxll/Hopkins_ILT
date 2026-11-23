from functools import partial

import torch
from hopkins.resist import hard_resist, soft_resist
from ilt.optimization.losses import boundary_weight_map, ilt_loss
from ilt.optimization.parameterization import FilteredMask
from ilt.utils.metrics import binary_iou
from tqdm import tqdm

from .regularizers import SPATIAL_PENALTIES


def _converged(history, config):
    recent = history[-config.early_stop_window:]
    start, end = recent[0]["loss"], recent[-1]["loss"]
    relative_loss = abs(start - end) / max(abs(start), 1e-12)
    ious = [row["iou"] for row in recent]
    return (relative_loss < config.early_stop_relative_loss
            and max(ious) - min(ious) < config.early_stop_iou_span
            and max(row["mask_change"] for row in recent) < config.early_stop_mask_change)


def optimize_mask(imaging, target, cfg, penalty, weight, *, label, on_step, show_progress,
                  wavelet_levels=3):
    """PixelILT schedule with a selectable differentiable mask penalty."""
    mask_model = FilteredMask(target, cfg.mask_filter_sigma_px, cfg.border_margin_px).to(target.device)
    optimizer = torch.optim.Adam(mask_model.parameters(), lr=cfg.learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.iterations, eta_min=cfg.minimum_learning_rate
    )
    weights = boundary_weight_map(target, cfg.edge_weight, cfg.boundary_radius_px)
    threshold = imaging.config.resist_threshold
    spatial_penalty = SPATIAL_PENALTIES[penalty]
    if penalty == "wavelet":
        spatial_penalty = partial(spatial_penalty, levels=wavelet_levels)
    hardening = max(1, round(cfg.iterations * cfg.hardening_fraction))
    history, previous, best_mask = [], None, None
    best_iou, best_step, convergence_hits = -1.0, 0, 0
    stop_reason = "maximum iterations reached"
    with tqdm(total=cfg.iterations, desc=label, unit="step", disable=not show_progress) as progress:
        for step in range(cfg.iterations):
            fraction = min(step / hardening, 1.0)
            mask_beta = cfg.mask_beta_start + fraction * (cfg.mask_beta_end - cfg.mask_beta_start)
            resist_beta = cfg.resist_beta_start + fraction * (cfg.resist_beta_end - cfg.resist_beta_start)
            optimizer.zero_grad(set_to_none=True)
            mask = mask_model(mask_beta)
            printed = soft_resist(imaging(mask), threshold, resist_beta)
            loss, parts = ilt_loss(mask, printed, target, weights,
                                  weight if penalty == "tv" else 0.0,
                                  cfg.binary_weight, cfg.area_weight)
            if penalty != "tv":
                regularizer = spatial_penalty(mask)
                loss = loss + weight * regularizer
                parts[penalty] = regularizer.detach()
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(mask_model.parameters(), cfg.gradient_clip))
            optimizer.step()
            schedule.step()
            completed = step + 1
            if on_step is not None:
                on_step(completed)
            progress.update()
            if step % cfg.log_every != 0 and completed != cfg.iterations:
                continue

            with torch.no_grad():
                hard = (mask_model(mask_beta) >= 0.5).float()
                iou = binary_iou(hard_resist(imaging(hard), threshold), target)
                change = 1.0 if previous is None else float((hard != previous).float().mean())
                previous = hard
                if iou > best_iou:
                    best_iou, best_step, best_mask = iou, completed, hard
            row = dict(step=completed, loss=float(loss.detach()), iou=iou, mask_change=change,
                       mask_beta=mask_beta, resist_beta=resist_beta, gradient_norm=grad_norm,
                       learning_rate=optimizer.param_groups[0]["lr"],
                       **{name: float(value) for name, value in parts.items()})
            history.append(row)
            if show_progress:
                progress.set_postfix(loss=f"{row['loss']:.5f}", iou=f"{iou:.5f}", refresh=False)
            if (cfg.early_stop and step >= max(cfg.early_stop_min_step, hardening)
                    and len(history) >= cfg.early_stop_window):
                convergence_hits = convergence_hits + 1 if _converged(history, cfg) else 0
                if convergence_hits >= cfg.early_stop_patience:
                    stop_reason = "early convergence: fixed-beta loss/IoU/mask-change plateau"
                    break
    return best_mask, history, dict(steps=completed, stop_reason=stop_reason, best_step=best_step)
