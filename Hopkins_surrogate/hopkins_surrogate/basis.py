from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.interpolate import BSpline


@dataclass(frozen=True)
class CorrectionBasis:
    """Precomputed, fixed basis for the frequency-domain kernel correction."""

    basis_dc: torch.Tensor      # [nsub, M], DC-projected 2D B-spline basis
    window: torch.Tensor        # [nsub], smooth radial support window
    signed_k: torch.Tensor      # [nsub, 2], signed frequency bins (ky, kx)
    dc_index: int
    kc: int
    n_fft: int
    n_basis: int


def signed_bin_index(sub: torch.Tensor, n_fft: int) -> torch.Tensor:
    """Map natural-order FFT indices to signed bins ``[-kc, kc]``."""
    sub = sub.to(torch.int64)
    threshold = (n_fft + 1) // 2
    return sub - n_fft * (sub >= threshold).to(torch.int64)


def cubic_bspline_matrix(u, n_basis: int, degree: int = 3) -> torch.Tensor:
    """Clamped uniform cubic B-spline design matrix ``[len(u), n_basis]``.

    Rows sum to one.  ``u`` is expected in ``[0, 1]``.
    """
    if n_basis < degree + 1:
        raise ValueError(f"n_basis must be >= {degree + 1}, got {n_basis}")
    n_internal = n_basis - degree - 1
    knots = (
        [0.0] * (degree + 1)
        + [(i + 1) / (n_internal + 1) for i in range(n_internal)]
        + [1.0] * (degree + 1)
    )
    if isinstance(u, torch.Tensor):
        u_np = u.detach().cpu().numpy()
    else:
        u_np = np.asarray(u)
    u_np = np.clip(u_np.astype(np.float64), 0.0, 1.0)
    design = BSpline.design_matrix(
        u_np, np.asarray(knots, dtype=np.float64), degree, extrapolate=True
    ).toarray()
    design = design.astype(np.float32)
    row_sum = design.sum(axis=1, keepdims=True)
    design = design / np.clip(row_sum, 1e-8, None)
    return torch.from_numpy(design)


def radial_window(signed_k: torch.Tensor, kc: int, inner: float = 0.75,
                  outer: float = 1.0) -> torch.Tensor:
    """Smooth disk window on the subgrid, normalized by ``kc``.

    ``W=1`` for ``r <= inner``, ``W=0`` for ``r >= outer``, smoothstep between.
    Corners of the square subgrid (``r > kc``) are suppressed.
    """
    if outer <= inner:
        raise ValueError("outer must be > inner")
    ky = signed_k[:, 0].to(torch.float32)
    kx = signed_k[:, 1].to(torch.float32)
    r = torch.sqrt(ky.square() + kx.square()) / float(kc)
    t = ((r - inner) / (outer - inner)).clamp(0.0, 1.0)
    return 1.0 - t.square() * (3.0 - 2.0 * t)


def build_correction_basis(
    sub_y: torch.Tensor,
    sub_x: torch.Tensor,
    n_fft: int,
    subgrid_shape: tuple[int, int],
    n_basis: int = 5,
    window_inner: float = 0.75,
    window_outer: float = 1.0,
    device: torch.device | str = "cpu",
) -> CorrectionBasis:
    """Build the DC-projected tensor-product B-spline basis on the subgrid."""
    if subgrid_shape[0] != subgrid_shape[1]:
        raise ValueError(f"subgrid must be square, got {subgrid_shape}")
    kc = (int(subgrid_shape[0]) - 1) // 2
    if kc < 1:
        raise ValueError(f"invalid subgrid shape {subgrid_shape}")

    sub_y = sub_y.detach().cpu().to(torch.int64)
    sub_x = sub_x.detach().cpu().to(torch.int64)
    nsub = int(sub_y.numel())
    if nsub != int(subgrid_shape[0]) * int(subgrid_shape[1]):
        raise ValueError(
            f"sub_y/sub_x size {nsub} != product {subgrid_shape}")

    signed_k = torch.stack(
        [signed_bin_index(sub_y, n_fft), signed_bin_index(sub_x, n_fft)],
        dim=1,
    )
    dc_index = kc * (2 * kc + 1) + kc
    if not torch.equal(signed_k[dc_index], torch.zeros(2, dtype=torch.int64)):
        raise ValueError("dc_index does not point to (0, 0)")

    u_y = (signed_k[:, 0] + kc).to(torch.float32) / float(2 * kc)
    u_x = (signed_k[:, 1] + kc).to(torch.float32) / float(2 * kc)
    a_y = cubic_bspline_matrix(u_y, n_basis)
    a_x = cubic_bspline_matrix(u_x, n_basis)
    basis_2d = (a_y[:, :, None] * a_x[:, None, :]).reshape(
        nsub, n_basis * n_basis)
    basis_dc = basis_2d - basis_2d[dc_index:dc_index + 1]
    window = radial_window(signed_k, kc, window_inner, window_outer)

    device = torch.device(device)
    return CorrectionBasis(
        basis_dc=basis_dc.to(device),
        window=window.to(device),
        signed_k=signed_k.to(device),
        dc_index=int(dc_index),
        kc=int(kc),
        n_fft=int(n_fft),
        n_basis=int(n_basis),
    )
