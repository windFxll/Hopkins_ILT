from torch import nn

from .abbe import AbbeImaging
from .config import OpticsConfig
from .socs import SocsImaging


def make_imaging(config: OpticsConfig, device, method: str = "abbe", *,
                 num_orders: int | None = None, kernel_batch_size: int = 8,
                 eigh_mode: str = "low_rank",
                 max_explicit_tcc_bytes: int = 512 * 1024**2) -> nn.Module:
    """Construct an imaging operator by method name (``abbe`` or ``socs``)."""
    method = str(method).lower()
    if method == "abbe":
        return AbbeImaging(config, device)
    if method == "socs":
        return SocsImaging(
            config, device, num_orders=num_orders,
            kernel_batch_size=kernel_batch_size, eigh_mode=eigh_mode,
            max_explicit_tcc_bytes=max_explicit_tcc_bytes)
    raise ValueError(f"unknown imaging method: {method!r}")
