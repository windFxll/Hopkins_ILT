"""Lightweight frequency-domain SOCS kernel correction.

The package wraps :class:`hopkins.socs.SocsImaging` and adds a low-dimensional
B-spline correction to each frequency-domain SOCS kernel.  The physical
eigenvalues and all optics parameters stay frozen.
"""

from .api import (DEFAULT_CHECKPOINT, HopkinsSurrogate, load_hybrid,
                  load_surrogate)
from .config import SurrogateConfig, TrainConfig
from .correction import KernelCorrection
from .imaging import CorrectedSocsImaging
from .residual import CorrectedResidualImaging, ResidualCNN

__all__ = [
    "CorrectedResidualImaging",
    "CorrectedSocsImaging",
    "DEFAULT_CHECKPOINT",
    "HopkinsSurrogate",
    "KernelCorrection",
    "ResidualCNN",
    "SurrogateConfig",
    "TrainConfig",
    "load_hybrid",
    "load_surrogate",
]
