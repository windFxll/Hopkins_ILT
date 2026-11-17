import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Hopkins"))
sys.path.insert(0, str(ROOT / "Hopkins_surrogate"))

from hopkins.config import OpticsConfig  # noqa: E402


@pytest.fixture
def small_config():
    return OpticsConfig(
        grid_size=96, pixel_size_nm=4.0,
        source_rings=2, source_points_per_ring=8,
        boundary_mode="periodic", aerial_scale=1.0)


@pytest.fixture
def device():
    return torch.device("cpu")
