from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizationConfig:
    iterations: int = 1000
    learning_rate: float = 0.08
    minimum_learning_rate: float = 0.002
    log_every: int = 5
    hardening_fraction: float = 0.30
    mask_beta_start: float = 1.8
    mask_beta_end: float = 10.0
    resist_beta_start: float = 8.0
    resist_beta_end: float = 40.0
    mask_filter_sigma_px: float = 1.25
    border_margin_px: int = 10
    edge_weight: float = 3.0
    boundary_radius_px: int = 2
    tv_weight: float = 5.0e-4
    binary_weight: float = 1.0e-3
    area_weight: float = 0.08
    gradient_clip: float = 5.0
    early_stop: bool = True
    early_stop_min_step: int = 350
    early_stop_window: int = 10
    early_stop_patience: int = 3
    early_stop_relative_loss: float = 5.0e-4
    early_stop_iou_span: float = 2.0e-4
    early_stop_mask_change: float = 2.0e-4
