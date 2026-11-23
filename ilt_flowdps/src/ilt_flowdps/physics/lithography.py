import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from hopkins.resist import soft_resist
from hopkins_surrogate import HopkinsSurrogate
from ilt.optimization.losses import boundary_weight_map, ilt_loss


def load_optics(checkpoint, device, *, allow_basis_mismatch=False):
    proxy = HopkinsSurrogate.from_pretrained(
        checkpoint, device=device, strict=not allow_basis_mismatch
    )
    # Identical optics, basis, normalization and K. Only the learned corrections differ.
    return {"socs": proxy.hybrid.kernel_model.base, "proxy": proxy}


def to_mask(state: torch.Tensor, encoding: str, polarity: str) -> torch.Tensor:
    value = state[0, 0]
    if encoding == "signed":
        value = (value + 1.0) * 0.5
    if polarity == "feature-low":
        value = 1.0 - value
    return value.clamp(0.0, 1.0)


def project_rf(state, encoding):
    """Project a mask-coordinate iterate to the checkpoint's valid value range."""
    return state.clamp(-1.0 if encoding == "signed" else 0.0, 1.0)


class LithographyObjective:
    def __init__(self, optics, target, config):
        self.optics = optics
        self.target = target
        self.config = config
        self.weights = boundary_weight_map(target, config.edge_weight, config.boundary_radius_px)

    def __call__(self, mask, beta):
        aerial = self.optics(mask)
        printed = soft_resist(aerial, self.optics.config.resist_threshold, beta)
        config = self.config
        loss, _ = ilt_loss(mask, printed, self.target, self.weights,
                          config.tv_weight, config.binary_weight, config.area_weight)
        return loss

    def print_loss(self, mask, beta):
        """Soft resist Bernoulli loss in aerial units for LPS and SGS updates."""
        logits = beta * (self.optics.config.resist_threshold - self.optics(mask))
        return binary_cross_entropy_with_logits(
            logits, self.target, weight=self.weights, reduction="mean"
        ) / beta
