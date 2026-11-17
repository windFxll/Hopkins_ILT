import math
import time
from dataclasses import dataclass

import torch

from ..config import OpticsConfig
from ..pupil import build_shifted_pupils


@dataclass(frozen=True)
class SocsEigenBasis:
    """Eigenpairs of the discretized Hopkins TCC, truncated to useful rank."""

    kernels: torch.Tensor          # [Kmax, nsub] complex64, CPU
    eigenvalues: torch.Tensor      # [Kmax] float32, CPU
    dc_power: torch.Tensor         # [Kmax] float32, CPU
    sub_y: torch.Tensor            # [nsub] int64, CPU
    sub_x: torch.Tensor            # [nsub] int64, CPU
    n_fft: int
    subgrid_shape: tuple
    dc_index: int
    rank: int
    clear_intensity: float
    eigenvalue_sum: float
    tcc_build_seconds: float
    eigh_seconds: float
    eigh_mode: str


def _subgrid_indices(config: OpticsConfig):
    n = config.fft_size
    df = 1.0 / (n * config.pixel_size_nm)
    fc = config.numerical_aperture / config.wavelength_nm
    kc = int(math.floor((1.0 + config.sigma_out) * fc / df))
    if kc < 1:
        raise ValueError(
            "illumination support is under-resolved: kc=%d < 1" % kc)
    if kc >= n // 2:
        raise ValueError(
            "illumination support exceeds the FFT band: kc=%d >= n/2=%d"
            % (kc, n // 2))

    half = torch.arange(-kc, kc + 1, dtype=torch.int64)
    ky, kx = torch.meshgrid(half, half, indexing="ij")
    ky = ky.reshape(-1)
    kx = kx.reshape(-1)
    dc_index = kc * (2 * kc + 1) + kc
    return (ky % n, kx % n, dc_index, (2 * kc + 1, 2 * kc + 1), n)


def _select_mode(mode: str, dtype: torch.dtype, nsub: int,
                 max_explicit_tcc_bytes: int) -> str:
    if mode not in ("low_rank", "explicit", "auto"):
        raise ValueError("unknown eigh_mode: %r" % (mode,))
    if mode != "auto":
        if mode == "explicit":
            required = nsub * nsub * torch.tensor([], dtype=dtype).element_size() * 2
            if required > max_explicit_tcc_bytes:
                raise ValueError(
                    "explicit TCC would need %.0f MB, above the %.0f MB budget; "
                    "use eigh_mode='low_rank'"
                    % (required / 2**20, max_explicit_tcc_bytes / 2**20))
        return mode
    required = nsub * nsub * torch.tensor([], dtype=dtype).element_size() * 2
    return "explicit" if required <= max_explicit_tcc_bytes else "low_rank"


def build_tcc_eigenbasis(
    config: OpticsConfig,
    device: torch.device,
    *,
    max_orders: int | None = None,
    eigh_mode: str = "low_rank",
    max_explicit_tcc_bytes: int = 512 * 1024**2,
    eigenvalue_tol: float = 1e-12,
) -> SocsEigenBasis:
    """Build the TCC eigenbasis used by SOCS imaging.

    TCC(i, j) = sum_s w_s P_s(f_i) P_s*(f_j), where P_s is the shifted pupil
    of source point s sampled on the frequency subgrid that can contribute
    (|f| <= (1 + sigma_out) * NA/lambda).  rank(TCC) <= source_count.

    ``low_rank`` decomposes the S x S Gram matrix A A^H and reconstructs the
    kernels exactly; ``explicit`` forms the full nsub x nsub TCC.  Both paths
    use torch.linalg.eigh.
    """
    device = torch.device(device)
    sub_y, sub_x, dc_index, subgrid_shape, n_fft = _subgrid_indices(config)
    nsub = sub_y.numel()

    pupils, weights, _ = build_shifted_pupils(config, device)
    sy = sub_y.to(pupils.device)
    sx = sub_x.to(pupils.device)
    P = pupils[:, sy, sx].detach().cpu()
    weights_cpu = weights.detach().cpu().to(torch.float64)

    if P.is_complex():
        dtype = torch.complex128
    else:
        dtype = torch.float64
    P = P.to(dtype)
    A = P * torch.sqrt(weights_cpu).to(dtype)[:, None]      # [S, nsub]

    mode = _select_mode(eigh_mode, dtype, nsub, max_explicit_tcc_bytes)

    if mode == "explicit":
        started = time.perf_counter()
        tcc = A.mH @ A                                      # [nsub, nsub]
        tcc = 0.5 * (tcc + tcc.mH)
        tcc_build_seconds = time.perf_counter() - started
        started = time.perf_counter()
        evals, evecs = torch.linalg.eigh(tcc)               # ascending
        eigh_seconds = time.perf_counter() - started
        del tcc
        evals = evals.flip(0)
        evecs = evecs.flip(1)
        threshold = max(float(evals[0]), 1e-30) * eigenvalue_tol
        rank = int((evals > threshold).sum())
        keep = rank if max_orders is None else min(rank, int(max_orders))
        kernels = evecs[:, :keep].T.contiguous()            # [keep, nsub]
        eigenvalues = evals[:keep]
        del evecs
    else:
        started = time.perf_counter()
        gram = A @ A.mH                                     # [S, S]
        gram = 0.5 * (gram + gram.mH)
        tcc_build_seconds = time.perf_counter() - started
        started = time.perf_counter()
        evals, evecs = torch.linalg.eigh(gram)              # ascending
        eigh_seconds = time.perf_counter() - started
        evals = evals.flip(0).clamp_min(0.0)
        evecs = evecs.flip(1)
        threshold = max(float(evals[0]), 1e-30) * eigenvalue_tol
        rank = int((evals > threshold).sum())
        keep = rank if max_orders is None else min(rank, int(max_orders))
        eigenvalues = evals[:keep]
        phi = A.mH @ evecs[:, :keep]                        # [nsub, keep]
        kernels = (phi / torch.sqrt(eigenvalues)[None, :]).T.contiguous()
        del phi, evecs

    kernels = kernels.to(torch.complex64)
    eigenvalues = eigenvalues.to(torch.float32)
    dc_power = eigenvalues * kernels[:, dc_index].abs().square()
    clear_intensity = float(dc_power.sum())
    eigenvalue_sum = float(evals.clamp_min(0.0).sum()) if mode == "explicit" \
        else float(evals.sum())

    return SocsEigenBasis(
        kernels=kernels,
        eigenvalues=eigenvalues,
        dc_power=dc_power.to(torch.float32),
        sub_y=sub_y,
        sub_x=sub_x,
        n_fft=n_fft,
        subgrid_shape=subgrid_shape,
        dc_index=dc_index,
        rank=rank,
        clear_intensity=clear_intensity,
        eigenvalue_sum=eigenvalue_sum,
        tcc_build_seconds=tcc_build_seconds,
        eigh_seconds=eigh_seconds,
        eigh_mode=mode,
    )
