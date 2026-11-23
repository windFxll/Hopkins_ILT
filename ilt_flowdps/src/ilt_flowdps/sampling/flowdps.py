from dataclasses import dataclass

import torch
from tqdm import tqdm

from ..physics.lithography import to_mask


@dataclass(frozen=True)
class FlowConfig:
    steps: int = 250
    inner_steps: int = 1
    scale: float = 1.0
    max_guidance_rms: float = 1.0
    min_clean_weight: float = 0.10
    hardening_fraction: float = 0.30
    resist_beta_start: float = 8.0
    resist_beta_end: float = 40.0
    encoding: str = "signed"
    polarity: str = "feature-high"
    log_every: int = 10


def _endpoint_descent(endpoint, objective, beta, time_weight, config):
    with torch.enable_grad():
        endpoint = endpoint.detach().requires_grad_(True)
        image = to_mask(endpoint, config.encoding, config.polarity)
        loss = objective(image, beta)
        (gradient,) = torch.autograd.grad(loss * image.numel(), endpoint)

    gradient = time_weight * gradient
    rms = gradient.square().mean().sqrt()
    gradient = gradient / (rms / config.max_guidance_rms).clamp_min(1.0)
    return gradient, loss.detach(), rms


@torch.no_grad()
def sample_flowdps(prior, objective, initial_noise, config, *, label, show_progress=True, on_step=None):
    """Euler FlowDPS with the clean-endpoint Jacobian approximation."""
    state = initial_noise.clone()
    dt = 1.0 / config.steps
    history = []
    with tqdm(total=config.steps, desc=label, unit="step", disable=not show_progress) as progress:
        for step in range(config.steps):
            time = step * dt
            next_time = (step + 1) * dt
            fraction = min(time / config.hardening_fraction, 1.0)
            beta = config.resist_beta_start + fraction * (config.resist_beta_end - config.resist_beta_start)
            velocity = prior(state, state.new_full((1,), time))
            clean = state + (1.0 - time) * velocity
            noise = state - time * velocity
            # FlowDPS Eq. 26 in noise-to-data time. Floor the clean coefficient
            # at the noise endpoint, where (1-t)/t^2 is singular.
            time_weight = (1.0 - time) / max(time, config.min_clean_weight) ** 2
            clean_step = dt * config.scale / (next_time * config.inner_steps)
            for _ in range(config.inner_steps):
                gradient, loss, rms = _endpoint_descent(clean, objective, beta, time_weight, config)
                clean = clean - clean_step * gradient
            state = next_time * clean + (1.0 - next_time) * noise

            completed = step + 1
            if completed == 1 or completed % config.log_every == 0 or completed == config.steps:
                loss_value, rms_value = torch.stack((loss, rms)).tolist()
                if not torch.isfinite(state).all().item():
                    raise FloatingPointError(f"{label}: non-finite RF state at step {completed}")
                history.append(dict(step=completed, time=time, loss=loss_value,
                                    guidance_rms_before_clip=rms_value))
                if show_progress:
                    progress.set_postfix(loss=f"{loss_value:.5f}", refresh=False)
            progress.update()
            if on_step is not None:
                on_step(completed)
    return to_mask(state, config.encoding, config.polarity), history
