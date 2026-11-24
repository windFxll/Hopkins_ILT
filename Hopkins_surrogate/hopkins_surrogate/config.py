from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from hopkins.config import OpticsConfig


@dataclass(frozen=True)
class SurrogateConfig:
    """Optics + correction + loss hyper-parameters.

    The training grid is 800x800 at 1 nm/px (periodic).  ``aerial_scale`` is
    the fixed photometric calibration ``s``; use :meth:`optics` to build the
    matching :class:`OpticsConfig`.
    """

    # optics
    grid_size: int = 800
    pixel_size_nm: float = 1.0
    wavelength_nm: float = 193.0
    numerical_aperture: float = 1.35
    sigma_in: float = 0.6
    sigma_out: float = 0.9
    source_rings: int = 12
    source_points_per_ring: int = 72
    defocus_nm: float = 0.0
    boundary_mode: str = "periodic"
    resist_threshold: float = 0.346585
    resist_blur_sigma_nm: float = 0.0
    aerial_scale: float | None = None

    # correction
    num_orders: int = 16
    basis_size: int = 5
    alpha_max: float = 0.05
    alpha_init: float = 0.02
    c_init_std: float = 0.0
    window_inner: float = 0.75
    window_outer: float = 1.0

    # loss / threshold (target units are raw .dat units)
    tau_dat: float = 0.346585
    band_width_norm: float = 0.08
    dice_temp_norm: float = 0.03
    lambda_bin: float = 0.3
    lambda_rel: float = 0.1
    # residual CNN loss weights (V1 full-field / multi-threshold).
    # Defaults reproduce the original residual loss exactly.
    lambda_band: float = 1.0
    lambda_dice: float = 0.3
    lambda_full: float = 0.0
    lambda_multi: float = 0.0
    multi_tau_factors: list[float] = field(default_factory=list)
    lambda_smooth: float = 0.0
    rel_norm_target: float = 0.05
    rel_norm_mean_max: float = 0.05
    rel_norm_max: float = 0.10

    def optics(self, scale: float | None = None) -> OpticsConfig:
        """Build the matching :class:`OpticsConfig`.

        ``scale`` overrides ``self.aerial_scale``; ``None`` means 1.0.
        """
        s = self.aerial_scale if scale is None else float(scale)
        if s is None:
            s = 1.0
        return OpticsConfig(
            grid_size=self.grid_size,
            pixel_size_nm=self.pixel_size_nm,
            wavelength_nm=self.wavelength_nm,
            numerical_aperture=self.numerical_aperture,
            sigma_in=self.sigma_in,
            sigma_out=self.sigma_out,
            source_rings=self.source_rings,
            source_points_per_ring=self.source_points_per_ring,
            defocus_nm=self.defocus_nm,
            boundary_mode=self.boundary_mode,
            resist_threshold=self.resist_threshold,
            resist_blur_sigma_nm=self.resist_blur_sigma_nm,
            aerial_scale=float(s),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SurrogateConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 4
    epochs: int = 50
    lr_c: float = 1e-2
    lr_alpha: float = 1e-3
    weight_decay_c: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 10
    warmup_steps: int = 200
    num_workers: int = 0
    seed: int = 0
    cd_every: int = 5
    lambda_res: float = 1e-3
    limit_train: int | None = None
    limit_val: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
