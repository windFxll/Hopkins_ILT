import torch
from torch import nn

from ..config import OpticsConfig
from ..resist.models import gaussian_blur2d
from .tcc import build_tcc_eigenbasis


class SocsImaging(nn.Module):
    """Differentiable SOCS imaging built from the Hopkins TCC eigenbasis.

    Input mask convention is identical to AbbeImaging: 1 = opaque dark
    feature, 0 = clear field.  The first ``num_orders`` eigenpairs of the
    discretized TCC are used; ``num_orders=None`` keeps the full numerical
    rank (bounded by the number of source points).
    """

    def __init__(self, config: OpticsConfig, device: torch.device | str,
                 num_orders: int | None = None, kernel_batch_size: int = 8,
                 eigh_mode: str = "low_rank",
                 max_explicit_tcc_bytes: int = 512 * 1024**2):
        super().__init__()
        self.config = config
        self.device = torch.device(device)
        self.kernel_batch_size = max(1, int(kernel_batch_size))
        with torch.no_grad():
            basis = build_tcc_eigenbasis(
                config, self.device, max_orders=None, eigh_mode=eigh_mode,
                max_explicit_tcc_bytes=max_explicit_tcc_bytes)

        self.max_orders = int(basis.kernels.shape[0])
        if num_orders is None:
            self.num_orders = self.max_orders
        else:
            self.num_orders = max(1, min(int(num_orders), self.max_orders))
        self.rank = basis.rank
        self.subgrid_shape = basis.subgrid_shape
        self.eigh_mode_used = basis.eigh_mode
        self.tcc_build_seconds = basis.tcc_build_seconds
        self.eigh_seconds = basis.eigh_seconds
        self.eigenvalue_sum = basis.eigenvalue_sum

        self.register_buffer("kernels", basis.kernels.to(self.device),
                             persistent=False)
        self.register_buffer("eigenvalues", basis.eigenvalues.to(self.device),
                             persistent=False)
        self.register_buffer("dc_power_cumulative",
                             torch.cumsum(basis.dc_power, dim=0).to(self.device),
                             persistent=False)
        self.register_buffer("sub_y", basis.sub_y.to(self.device),
                             persistent=False)
        self.register_buffer("sub_x", basis.sub_x.to(self.device),
                             persistent=False)
        self.register_buffer(
            "clear_intensity",
            torch.tensor(basis.clear_intensity, dtype=torch.float32,
                         device=self.device),
            persistent=False)

    def energy_fraction(self, num_orders: int | None = None) -> float:
        orders = self.num_orders if num_orders is None else int(num_orders)
        orders = max(1, min(orders, self.max_orders))
        return float(self.eigenvalues[:orders].sum() / self.eigenvalue_sum)

    def truncated_clear_intensity(self, num_orders: int | None = None) -> float:
        orders = self.num_orders if num_orders is None else int(num_orders)
        orders = max(1, min(orders, self.max_orders))
        return float(self.dc_power_cumulative[orders - 1])

    def set_num_orders(self, num_orders: int) -> None:
        self.num_orders = max(1, min(int(num_orders), self.max_orders))

    def forward(self, mask: torch.Tensor,
                num_orders: int | None = None,
                kernels: torch.Tensor | None = None) -> torch.Tensor:
        """Compute the normalized aerial image.

        ``kernels`` optionally overrides the first ``orders`` frequency-domain
        SOCS kernels; it must have shape ``[Q, nsub]`` and complex64 dtype
        (matching ``self.kernels``).  Normalization still uses the original
        ``dc_power_cumulative``.  Passing ``None`` keeps the original path
        bit-for-bit.
        """
        if mask.ndim != 2 or mask.shape != (self.config.grid_size,
                                            self.config.grid_size):
            raise ValueError(
                f"expected mask [{self.config.grid_size}, "
                f"{self.config.grid_size}]")

        orders = self.num_orders if num_orders is None else int(num_orders)
        orders = max(1, min(orders, self.max_orders))

        kern_all = self.kernels
        if kernels is not None:
            if kernels.ndim != 2:
                raise ValueError("kernels must be 2D [Q, nsub]")
            if kernels.shape[1] != self.kernels.shape[1]:
                raise ValueError(
                    f"kernels second dim {kernels.shape[1]} != "
                    f"nsub {self.kernels.shape[1]}")
            if kernels.shape[0] < orders:
                raise ValueError(
                    f"kernels has {kernels.shape[0]} rows, need >= {orders}")
            if (kernels.device != self.kernels.device
                    or kernels.dtype != self.kernels.dtype):
                kernels = kernels.to(device=self.kernels.device,
                                     dtype=self.kernels.dtype)
            kern_all = kernels

        transmission = 1.0 - mask.to(dtype=torch.float32)
        n = self.config.grid_size
        n_fft = self.config.fft_size
        if n_fft != n:
            total = n_fft - n
            left = total // 2
            right = total - left
            transmission = torch.nn.functional.pad(
                transmission[None, None],
                (left, right, left, right), value=1.0)[0, 0]

        spectrum = torch.fft.fft2(torch.fft.ifftshift(transmission))
        spectrum_sub = spectrum[self.sub_y, self.sub_x]

        intensity = torch.zeros((n_fft, n_fft), device=mask.device,
                                dtype=torch.float32)
        for start in range(0, orders, self.kernel_batch_size):
            stop = min(start + self.kernel_batch_size, orders)
            kern = kern_all[start:stop]
            lam = self.eigenvalues[start:stop]
            spec = torch.zeros((stop - start, n_fft, n_fft),
                               device=mask.device, dtype=torch.complex64)
            spec[:, self.sub_y, self.sub_x] = spectrum_sub[None] * kern
            field = torch.fft.fftshift(
                torch.fft.ifft2(spec, dim=(-2, -1)), dim=(-2, -1))
            power = field.real.square() + field.imag.square()
            intensity = intensity + torch.einsum("k,kij->ij", lam, power)

        if n_fft != n:
            offset = (n_fft - n) // 2
            intensity = intensity[offset:offset + n, offset:offset + n]
        intensity = intensity / self.dc_power_cumulative[orders - 1].clamp_min(
            1e-12)

        sigma_px = self.config.resist_blur_sigma_nm / self.config.pixel_size_nm
        if sigma_px > 0:
            intensity = gaussian_blur2d(
                intensity, sigma_px,
                circular=self.config.boundary_mode == "periodic")
        if self.config.aerial_scale != 1.0:
            intensity = intensity * float(self.config.aerial_scale)
        return intensity
