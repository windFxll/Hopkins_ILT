"""High-level inference API for the K=8 lithography surrogate.

The default model is a frozen K=8 SOCS imaging model with a bounded residual
CNN; checkpoints may additionally carry a low-rank frequency-domain kernel
correction.  This module hides the checkpoint plumbing and exposes a small,
ILT-friendly callable:

    model = HopkinsSurrogate.from_pretrained(device="cuda:0")
    aerial = model(mask)                  # differentiable w.r.t. mask
    out = model.predict(mask)             # aerial + clear/printed binaries

Conventions (match ``Hopkins`` and ``ILT``):

- mask: ``[H, W]`` or ``[B, H, W]``, 1 = opaque feature, 0 = clear.
- grid: 800 x 800 at 1 nm/px, periodic boundary.
- aerial: same shape as the mask, in raw ``.dat`` intensity units
  (the checkpoint's ``aerial_scale`` is already included).
- threshold: taken from the checkpoint; ``clear = aerial > threshold``
  (1 = clear), ``printed = aerial < threshold`` (1 = feature, matching ILT
  targets).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .checkpoint import build_model_from_checkpoint
from .residual import CorrectedResidualImaging, ResidualCNN

DEFAULT_CHECKPOINT = (Path(__file__).resolve().parents[1]
                      / "models" / "k8_cnn_default.pt")


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_hybrid(checkpoint: str | Path = DEFAULT_CHECKPOINT,
                device: str | torch.device | None = None,
                strict: bool = True
                ) -> tuple[CorrectedResidualImaging, dict[str, Any]]:
    """Load the frozen hybrid (kernel correction + residual CNN).

    Returns the raw :class:`CorrectedResidualImaging` module and the checkpoint
    payload.  Prefer :meth:`HopkinsSurrogate.from_pretrained` for a stable API.
    """
    device = _resolve_device(device)
    ckpt = torch.load(Path(checkpoint), map_location="cpu",
                      weights_only=False)
    if "residual_state" not in ckpt or "kernel_ckpt" not in ckpt:
        raise ValueError(
            "checkpoint is not a residual-CNN checkpoint (missing "
            "'residual_state' or 'kernel_ckpt'); pass a hybrid surrogate "
            "checkpoint such as k8_cnn_default.pt")
    kernel_model, _ = build_model_from_checkpoint(
        ckpt["kernel_ckpt"], device, strict=strict)
    residual = ResidualCNN(
        hidden=int(ckpt["hidden"]),
        delta_max=float(ckpt["delta_max"]),
        downscale=int(ckpt["downscale"]),
    ).to(device)
    residual.load_state_dict(ckpt["residual_state"])
    # detach_socs=False keeps the SOCS path differentiable w.r.t. the mask,
    # which is required for ILT.  All network parameters stay frozen.
    hybrid = CorrectedResidualImaging(
        kernel_model, residual, freeze_kernel=True, detach_socs=False)
    hybrid.eval()
    for param in hybrid.parameters():
        param.requires_grad_(False)
    return hybrid, ckpt


class HopkinsSurrogate(nn.Module):
    """Inference/ILT wrapper around the final K=8 surrogate model.

    The module has no trainable parameters.  ``forward`` is differentiable
    with respect to the input mask, so it can be used directly as the imaging
    model of :class:`ilt.optimization.PixelILT`.
    """

    def __init__(self, hybrid: CorrectedResidualImaging, ckpt: dict[str, Any],
                 checkpoint_path: str | Path | None = None):
        super().__init__()
        self.hybrid = hybrid
        self.checkpoint = ckpt
        self.checkpoint_path = (str(Path(checkpoint_path).resolve())
                                if checkpoint_path is not None else None)
        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)

    # -- construction -----------------------------------------------------
    @classmethod
    def from_pretrained(cls, checkpoint: str | Path = DEFAULT_CHECKPOINT,
                        device: str | torch.device | None = None,
                        strict: bool = True) -> "HopkinsSurrogate":
        hybrid, ckpt = load_hybrid(checkpoint, device=device, strict=strict)
        return cls(hybrid, ckpt, checkpoint_path=checkpoint)

    # -- metadata ---------------------------------------------------------
    @property
    def config(self):
        """The underlying :class:`hopkins.config.OpticsConfig`."""
        return self.hybrid.config

    @property
    def threshold(self) -> float:
        """Binary threshold in raw ``.dat`` units."""
        return float(self.config.resist_threshold)

    @property
    def tau(self) -> float:
        """Alias for :attr:`threshold`."""
        return self.threshold

    @property
    def grid_size(self) -> int:
        return int(self.config.grid_size)

    @property
    def pixel_size_nm(self) -> float:
        return float(self.config.pixel_size_nm)

    @property
    def aerial_scale(self) -> float:
        return float(self.config.aerial_scale)

    @property
    def num_orders(self) -> int:
        return int(self.hybrid.kernel_model.num_orders)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # -- inference --------------------------------------------------------
    def _prepare(self, mask: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(mask, np.ndarray):
            mask = torch.as_tensor(mask, dtype=torch.float32,
                                   device=self.device)
        if not torch.is_tensor(mask):
            raise TypeError(
                f"mask must be a torch.Tensor or numpy.ndarray, got "
                f"{type(mask).__name__}")
        if mask.ndim not in (2, 3):
            raise ValueError(
                f"mask must be [H, W] or [B, H, W], got shape "
                f"{tuple(mask.shape)}")
        if tuple(mask.shape[-2:]) != (self.grid_size, self.grid_size):
            raise ValueError(
                f"mask spatial shape must be {self.grid_size}x{self.grid_size}"
                f" at {self.pixel_size_nm} nm/px, got "
                f"{tuple(mask.shape[-2:])}")
        return mask.to(device=self.device, dtype=torch.float32)

    def forward(self, mask: torch.Tensor | np.ndarray,
                use_correction: bool = True,
                use_residual: bool = True) -> torch.Tensor:
        """Return the aerial intensity in raw ``.dat`` units.

        Gradients flow back to ``mask``; all model parameters are frozen.
        A 3-D input is processed as a batch (a Python loop, not vectorized).
        """
        mask = self._prepare(mask)
        if mask.ndim == 2:
            return self.hybrid(mask, use_correction=use_correction,
                               use_residual=use_residual)
        return torch.stack([
            self.hybrid(item, use_correction=use_correction,
                        use_residual=use_residual)
            for item in mask
        ])

    @torch.no_grad()
    def predict(self, mask: torch.Tensor | np.ndarray,
                use_correction: bool = True,
                use_residual: bool = True) -> dict[str, Any]:
        """Inference-only convenience: aerial plus clear/printed binaries.

        Returns a dict with:

        - ``aerial``: float tensor in raw ``.dat`` units;
        - ``clear``: bool tensor, 1 = clear (``aerial > threshold``);
        - ``printed``: bool tensor, 1 = feature (``aerial < threshold``);
        - ``threshold``: float, the checkpoint's resist threshold.
        """
        aerial = self.forward(mask, use_correction=use_correction,
                              use_residual=use_residual)
        return {
            "aerial": aerial,
            "clear": aerial > self.threshold,
            "printed": aerial < self.threshold,
            "threshold": self.threshold,
        }

    def extra_repr(self) -> str:
        path = self.checkpoint_path or "in-memory"
        n_params = sum(p.numel() for p in self.parameters())
        return (f"checkpoint={path}, grid={self.grid_size}, "
                f"pixel_size_nm={self.pixel_size_nm}, "
                f"num_orders={self.num_orders}, "
                f"threshold={self.threshold}, params={n_params}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"


def load_surrogate(checkpoint: str | Path = DEFAULT_CHECKPOINT,
                   device: str | torch.device | None = None,
                   strict: bool = True) -> HopkinsSurrogate:
    """Convenience wrapper for :meth:`HopkinsSurrogate.from_pretrained`."""
    return HopkinsSurrogate.from_pretrained(
        checkpoint, device=device, strict=strict)


__all__ = ["DEFAULT_CHECKPOINT", "HopkinsSurrogate", "load_hybrid",
           "load_surrogate"]
