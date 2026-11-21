import torch

from ..physics.lithography import to_mask


def optical_gradient(state, objective, beta, config):
    """Differentiate the summed optical energy in RF mask coordinates."""
    with torch.enable_grad():
        variable = state.detach().requires_grad_(True)
        mask = to_mask(variable, config.encoding, config.polarity)
        loss = objective.print_loss(mask, beta)
        (gradient,) = torch.autograd.grad(loss * mask.numel(), variable)
    return gradient, loss.detach()


def randn_like(state, generator):
    return torch.randn(state.shape, dtype=state.dtype, device=state.device, generator=generator)


def resist_beta(time, config):
    fraction = min(time / config.hardening_fraction, 1.0)
    return config.resist_beta_start + fraction * (config.resist_beta_end - config.resist_beta_start)


def log_sample(history, step, steps, every, state, loss, progress, on_step, *, label, **fields):
    completed = step + 1
    if completed == 1 or completed % every == 0 or completed == steps:
        if not torch.isfinite(state).all().item() or not torch.isfinite(loss).item():
            raise FloatingPointError(f"{label}: non-finite result at step {completed}")
        loss_value = float(loss)
        history.append(dict(step=completed, loss=loss_value, **fields))
        if not progress.disable:
            progress.set_postfix(loss=f"{loss_value:.5f}", refresh=False)
    progress.update()
    if on_step is not None:
        on_step(completed)
