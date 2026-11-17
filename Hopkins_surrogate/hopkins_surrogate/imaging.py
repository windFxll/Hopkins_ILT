from __future__ import annotations

import torch
from torch import nn

from hopkins.config import OpticsConfig
from hopkins.socs import SocsImaging

from .basis import build_correction_basis
from .correction import KernelCorrection


class CorrectedSocsImaging(nn.Module):
    """SOCS imaging with a trainable low-rank frequency-domain correction.

    The wrapper keeps the original :class:`SocsImaging` untouched.  When the
    correction is disabled (or ``use_correction=False``), ``forward`` calls the
    base model without overriding kernels, so the output is bit-for-bit
    identical to the original SOCS path.
    """

    def __init__(self, config: OpticsConfig, device,
                 num_orders: int = 16, basis_size: int = 5,
                 alpha_max: float = 0.05, alpha_init: float = 1e-3,
                 c_init_std: float = 1e-3, correction_enabled: bool = True,
                 kernel_batch_size: int = 8, eigh_mode: str = "low_rank",
                 max_explicit_tcc_bytes: int = 512 * 1024**2,
                 window_inner: float = 0.75, window_outer: float = 1.0,
                 seed: int | None = None):
        super().__init__()
        self.config = config
        self.base = SocsImaging(
            config, device, num_orders=num_orders,
            kernel_batch_size=kernel_batch_size, eigh_mode=eigh_mode,
            max_explicit_tcc_bytes=max_explicit_tcc_bytes)
        self.num_orders = int(self.base.num_orders)
        self.rank = int(self.base.rank)
        self.subgrid_shape = self.base.subgrid_shape

        basis = build_correction_basis(
            self.base.sub_y, self.base.sub_x, config.fft_size,
            self.base.subgrid_shape, n_basis=basis_size,
            window_inner=window_inner, window_outer=window_outer,
            device=self.base.device)
        self.correction = KernelCorrection(
            basis, self.num_orders, alpha_max=alpha_max,
            alpha_init=alpha_init, c_init_std=c_init_std, seed=seed)
        self.correction.to(self.base.device)
        self.correction_enabled = bool(correction_enabled)
        self.correction.set_enabled(self.correction_enabled)

    def set_correction_enabled(self, enabled: bool) -> None:
        self.correction_enabled = bool(enabled)
        self.correction.set_enabled(self.correction_enabled)

    def corrected_kernels(self, num_orders: int | None = None) -> torch.Tensor:
        q = self.num_orders if num_orders is None else int(num_orders)
        q = max(1, min(q, self.num_orders))
        return self.correction.corrected_kernels(self.base.kernels, q)

    def forward(self, mask: torch.Tensor,
                num_orders: int | None = None,
                use_correction: bool | None = None) -> torch.Tensor:
        active = (self.correction_enabled if use_correction is None
                  else bool(use_correction))
        if mask.ndim == 3:
            return torch.stack([
                self.forward(m, num_orders=num_orders,
                             use_correction=use_correction)
                for m in mask], dim=0)
        if mask.ndim != 2:
            raise ValueError("expected mask [N, N] or [B, N, N]")
        if not active:
            return self.base(mask, num_orders=num_orders)
        q = self.num_orders if num_orders is None else int(num_orders)
        q = max(1, min(q, self.num_orders))
        kernels = self.correction.corrected_kernels(self.base.kernels, q)
        return self.base(mask, num_orders=q, kernels=kernels)
