from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from .imaging import CorrectedSocsImaging


class ResidualCNN(nn.Module):
    """Small mask-conditioned residual head.

    The head runs on a 1/downscale grid, predicts a bounded residual, and
    upsamples it back to the full grid.  With the final layer zero-initialized,
    the initial output is exactly zero, so the hybrid model starts from the
    frozen kernel-only prediction.
    """

    def __init__(self, in_channels: int = 3, hidden: int = 24,
                 delta_max: float = 0.10, downscale: int = 4):
        super().__init__()
        self.delta_max = float(delta_max)
        self.downscale = int(downscale)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=4, dilation=4),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, mask: torch.Tensor,
                intensity_norm: torch.Tensor) -> torch.Tensor:
        if mask.ndim != 2 or intensity_norm.ndim != 2:
            raise ValueError("expected 2D mask and intensity")
        channels = torch.stack(
            [mask.float(), 1.0 - mask.float(), intensity_norm.float()],
            dim=0)[None]
        if self.downscale > 1:
            channels = F.interpolate(
                channels, scale_factor=1.0 / self.downscale, mode="area")
        y = self.net(channels)
        if self.downscale > 1:
            y = F.interpolate(
                y, size=mask.shape[-2:], mode="bilinear",
                align_corners=False)
        return torch.tanh(y[0, 0]) * self.delta_max


class CorrectedResidualImaging(nn.Module):
    """Frozen kernel-corrected SOCS plus a trainable bounded residual head."""

    def __init__(self, kernel_model: CorrectedSocsImaging,
                 residual: ResidualCNN, freeze_kernel: bool = True,
                 detach_socs: bool = True):
        super().__init__()
        self.kernel_model = kernel_model
        self.residual = residual
        self.config = kernel_model.config
        self.scale = float(kernel_model.config.aerial_scale)
        self.detach_socs = bool(detach_socs)
        if freeze_kernel:
            for param in self.kernel_model.parameters():
                param.requires_grad_(False)

    def forward(self, mask: torch.Tensor,
                use_correction: bool = True,
                use_residual: bool = True) -> torch.Tensor:
        if not use_residual:
            return self.kernel_model(mask, use_correction=use_correction)
        context = torch.no_grad() if self.detach_socs else nullcontext()
        with context:
            socs = self.kernel_model(
                mask, use_correction=use_correction)
        delta = self.residual(mask, socs / self.scale)
        return (socs + delta).clamp_min(0.0)

    def residual_delta(self, mask: torch.Tensor,
                       use_correction: bool = True) -> torch.Tensor:
        with torch.no_grad():
            socs = self.kernel_model(
                mask, use_correction=use_correction)
        return self.residual(mask, socs / self.scale)
