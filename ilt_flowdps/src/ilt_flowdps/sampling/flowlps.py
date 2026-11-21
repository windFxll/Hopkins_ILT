from dataclasses import dataclass
import math

import torch
from tqdm import tqdm

from ..physics.lithography import project_rf, to_mask
from .optical import log_sample, optical_gradient, randn_like, resist_beta


@dataclass(frozen=True)
class LPSConfig:
    langevin_steps: int = 1
    proximal_steps: int = 4
    langevin_lr: float = 0.002
    proximal_lr: float = 0.2
    proximal_anchor: float = 0.25
    min_prior_std: float = 0.10


@torch.no_grad()
def sample_flowlps(prior, objective, initial_noise, flow, options, *, generator, label,
                   show_progress=True, on_step=None):
    """Pixel-space FlowLPS with bounded mask coordinates and a resist likelihood."""
    state = initial_noise.clone()
    history = []
    with tqdm(total=flow.steps, desc=label, unit="step", disable=not show_progress) as progress:
        for step in range(flow.steps):
            time = step / flow.steps
            next_time = (step + 1) / flow.steps
            beta = resist_beta(time, flow)
            velocity = prior(state, state.new_full((1,), time))
            anchor = project_rf(state + (1.0 - time) * velocity, flow.encoding)
            noise = state - time * velocity

            clean = anchor
            variance = max(1.0 - time, options.min_prior_std) ** 2
            eta = options.langevin_lr * variance
            for _ in range(options.langevin_steps):
                gradient, loss = optical_gradient(clean, objective, beta, flow)
                drift = gradient + (clean - anchor) / variance
                perturbation = math.sqrt(2.0 * eta) * randn_like(clean, generator)
                clean = project_rf(clean - eta * drift + perturbation, flow.encoding)

            local_anchor = clean
            for _ in range(options.proximal_steps):
                gradient, loss = optical_gradient(clean, objective, beta, flow)
                drift = gradient + options.proximal_anchor * (clean - local_anchor)
                clean = project_rf(clean - options.proximal_lr * drift, flow.encoding)

            # pCN refresh in the noise coordinate, followed by the linear flow update.
            if next_time < 1.0:
                noise = (math.sqrt(next_time) * noise
                         + math.sqrt(1.0 - next_time) * randn_like(noise, generator))
            state = next_time * clean + (1.0 - next_time) * noise
            log_sample(history, step, flow.steps, flow.log_every, state, loss,
                       progress, on_step, label=label, time=time, resist_beta=beta)
    return to_mask(state, flow.encoding, flow.polarity), history
