from dataclasses import dataclass
import math

import torch
from tqdm import tqdm

from ..physics.lithography import project_rf, to_mask
from .optical import log_sample, optical_gradient, randn_like, resist_beta


@dataclass(frozen=True)
class SGSConfig:
    steps: int = 40
    likelihood_steps: int = 8
    prior_steps: int = 4
    likelihood_lr: float = 0.02
    rho_start: float = 1.0
    rho_min: float = 0.12
    rho_decay: float = 0.90
    diffusion: float = 1.0


def _prior_step(prior, observation, rho, options, generator):
    # x_t = t*x_clean + (1-t)*noise. Match the incoming perturbation
    # rho to (1-t)/t, then integrate from that exact t to the clean endpoint.
    start = 1.0 / (1.0 + rho)
    state = start * observation
    dt = (1.0 - start) / options.prior_steps
    for index in range(options.prior_steps):
        time = start + index * dt
        velocity = prior(state, state.new_full((1,), time))
        noise = state - time * velocity
        # score = -noise / (1-t). Use w(t)=2*d*(1-t) in the
        # SI reverse SDE, expressed in this network's noise-to-data time.
        drift = velocity - options.diffusion * noise
        diffusion_std = math.sqrt(2.0 * options.diffusion * (1.0 - time) * dt)
        state = state + dt * drift + diffusion_std * randn_like(state, generator)
    return state


@torch.no_grad()
def sample_flowsgs(prior, objective, initial_noise, flow, options, *, generator, label,
                   show_progress=True, on_step=None):
    """Split Gibbs with a projected optical Langevin step and an RF prior SDE."""
    state = initial_noise.clone()
    data = project_rf(state, flow.encoding)
    history = []
    with tqdm(total=options.steps, desc=label, unit="step", disable=not show_progress) as progress:
        for step in range(options.steps):
            rho = max(options.rho_min, options.rho_start * options.rho_decay ** step)
            beta = resist_beta(step / max(options.steps - 1, 1), flow)
            variance = rho ** 2
            eta = min(options.likelihood_lr, 0.2 * variance)
            for _ in range(options.likelihood_steps):
                gradient, loss = optical_gradient(data, objective, beta, flow)
                drift = gradient + (data - state) / variance
                perturbation = math.sqrt(2.0 * eta) * randn_like(data, generator)
                data = project_rf(data - eta * drift + perturbation, flow.encoding)
            state = _prior_step(prior, data, rho, options, generator)
            log_sample(history, step, options.steps, flow.log_every, state, loss,
                       progress, on_step, label=label, rho=rho, resist_beta=beta)
    return to_mask(state, flow.encoding, flow.polarity), history
