from __future__ import annotations

import math

import torch
from torch import nn

from .basis import CorrectionBasis


class KernelCorrection(nn.Module):
    """Low-dimensional complex correction of SOCS frequency-domain kernels.

    ``delta_phi(q) = alpha * W * (B2_dc @ c_q)`` with ``delta_phi(0) = 0``.
    ``c_q`` is stored as separate real/imaginary parameters for optimizer
    compatibility.
    """

    def __init__(self, basis: CorrectionBasis, num_orders: int,
                 alpha_max: float = 0.05, alpha_init: float = 1e-3,
                 c_init_std: float = 1e-3, seed: int | None = None):
        super().__init__()
        if not 0.0 < alpha_init < alpha_max:
            raise ValueError("require 0 < alpha_init < alpha_max")
        self.num_orders = int(num_orders)
        self.n_basis = int(basis.n_basis)
        self.n_coeff = self.n_basis * self.n_basis
        self.alpha_max = float(alpha_max)
        self.register_buffer("basis_dc", basis.basis_dc, persistent=False)
        self.register_buffer("window", basis.window, persistent=False)
        self.register_buffer("signed_k", basis.signed_k, persistent=False)
        self.c_real = nn.Parameter(
            torch.empty(self.num_orders, self.n_coeff))
        self.c_imag = nn.Parameter(
            torch.empty(self.num_orders, self.n_coeff))
        self.alpha_logit = nn.Parameter(torch.tensor(
            math.log(alpha_init / (alpha_max - alpha_init)),
            dtype=torch.float32))
        self.reset_parameters(c_init_std, seed)
        self.enabled = True

    def reset_parameters(self, std: float, seed: int | None = None) -> None:
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(int(seed))
        with torch.no_grad():
            self.c_real.normal_(0.0, std, generator=generator)
            self.c_imag.normal_(0.0, std, generator=generator)

    @property
    def alpha(self) -> torch.Tensor:
        return self.alpha_max * torch.sigmoid(self.alpha_logit)

    def complex_coefficients(self, orders: int | None = None) -> torch.Tensor:
        q = self.num_orders if orders is None else int(orders)
        q = max(1, min(q, self.num_orders))
        return torch.complex(self.c_real[:q], self.c_imag[:q])

    def delta_phi(self, base_kernels: torch.Tensor,
                  orders: int | None = None) -> torch.Tensor:
        """Return ``ΔΦ`` with shape ``[q, nsub]`` (complex64)."""
        q = self.num_orders if orders is None else int(orders)
        q = max(1, min(q, self.num_orders))
        coeff = self.complex_coefficients(q)
        basis = self.basis_dc.to(dtype=coeff.dtype)
        window = self.window.to(dtype=coeff.dtype)
        delta = (coeff @ basis.T) * window[None, :]
        return self.alpha * delta

    def corrected_kernels(self, base_kernels: torch.Tensor,
                          orders: int | None = None) -> torch.Tensor:
        q = self.num_orders if orders is None else int(orders)
        q = max(1, min(q, self.num_orders))
        base = base_kernels[:q]
        if not self.enabled:
            return base
        return base + self.delta_phi(base, q)

    def relative_norm(self, base_kernels: torch.Tensor,
                      orders: int | None = None) -> torch.Tensor:
        q = self.num_orders if orders is None else int(orders)
        q = max(1, min(q, self.num_orders))
        base = base_kernels[:q]
        delta = self.delta_phi(base, q)
        return delta.norm(dim=1) / base.norm(dim=1).clamp_min(1e-8)

    def coefficient_smoothness(self) -> torch.Tensor:
        """Absolute second-difference penalty on the coefficient grid.

        Unlike a scale-free ratio, this vanishes (with zero gradient) at
        ``c=0`` and grows quadratically with the correction magnitude.
        """
        cr = self.c_real.view(self.num_orders, self.n_basis, self.n_basis)
        ci = self.c_imag.view(self.num_orders, self.n_basis, self.n_basis)

        def _d2(x: torch.Tensor) -> torch.Tensor:
            a = x[:, 2:, :] - 2.0 * x[:, 1:-1, :] + x[:, :-2, :]
            b = x[:, :, 2:] - 2.0 * x[:, :, 1:-1] + x[:, :, :-2]
            return a.square().mean() + b.square().mean()

        return _d2(cr) + _d2(ci)

    def smoothness_loss(self) -> torch.Tensor:
        """Backward-compatible alias for :meth:`coefficient_smoothness`."""
        return self.coefficient_smoothness()

    @torch.no_grad()
    def project_relative_norm(self, base_kernels: torch.Tensor,
                              mean_max: float = 0.05,
                              mode_max: float = 0.10,
                              orders: int | None = None) -> None:
        """Bound the relative correction norm after an optimizer step.

        First caps each mode at ``mode_max``, then (if needed) scales all modes
        so the mean is at most ``mean_max``.  The correction therefore stays a
        small, interpretable compensation of the physical kernels.
        """
        q = self.num_orders if orders is None else int(orders)
        q = max(1, min(q, self.num_orders))
        rel = self.relative_norm(base_kernels, q)
        scale_mode = (float(mode_max) / rel.clamp_min(1e-12)).clamp(max=1.0)
        self.c_real[:q].mul_(scale_mode[:, None])
        self.c_imag[:q].mul_(scale_mode[:, None])
        rel = rel * scale_mode
        scale_mean = (float(mean_max) / rel.mean().clamp_min(1e-12)).clamp(
            max=1.0)
        self.c_real[:q].mul_(scale_mean)
        self.c_imag[:q].mul_(scale_mean)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
