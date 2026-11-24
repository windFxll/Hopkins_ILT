from dataclasses import dataclass


@dataclass(frozen=True)
class OpticsConfig:
    """Optical model configuration.

    Defaults use the current dry-air illumination (NA=0.93, sigma=0.6-0.9,
    864 source points).  The 256x256 @ 4 nm grid is a lightweight demo
    sampling; the default surrogate checkpoint uses 800x800 @ 1 nm.
    ``aerial_scale`` stays 1.0 so the clear field is normalized to 1; the
    surrogate checkpoint uses 0.85107421875 to match raw S-Litho .dat units,
    and its ``resist_threshold`` is calibrated for that scale.
    """

    grid_size: int = 256
    pixel_size_nm: float = 4.0
    wavelength_nm: float = 193.0
    numerical_aperture: float = 0.93
    sigma_in: float = 0.6
    sigma_out: float = 0.9
    source_rings: int = 12
    source_points_per_ring: int = 72
    defocus_nm: float = 0.0
    boundary_mode: str = "periodic"
    resist_threshold: float = 0.3357889756677475
    resist_blur_sigma_nm: float = 0.0
    source_batch_size: int = 6
    aerial_scale: float = 1.0

    @property
    def fft_size(self) -> int:
        return 2 * self.grid_size if self.boundary_mode == "isolated" else self.grid_size

    @property
    def source_count(self) -> int:
        return self.source_rings * self.source_points_per_ring
