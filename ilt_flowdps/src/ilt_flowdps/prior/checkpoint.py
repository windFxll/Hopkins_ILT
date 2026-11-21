from pathlib import Path

import torch

from .unet import UNet2D


DEFAULT_RF_CHECKPOINT = Path(__file__).resolve().parents[3] / "checkpoints" / "rf_latest.pt"


def load_prior(path: Path, device: torch.device, grid_size: int):
    """Load ILT RF weights, using EMA when available."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("objective") != "rectified_flow" or payload.get("conditional") is not False:
        raise ValueError("Expected an unconditional ILT rectified_flow checkpoint")
    if payload.get("model_arch") != UNet2D.architecture:
        raise ValueError(f"Unsupported ILT model_arch: {payload.get('model_arch')!r}")
    if payload.get("image_size") != grid_size or grid_size % 16:
        raise ValueError(f"RF grid {payload.get('image_size')} does not match optical grid {grid_size}")

    source = "ema" if payload.get("ema") else "model"
    model = UNet2D(base=int(payload["base"]))
    model.load_state_dict(payload[source], strict=True)
    metadata = {key: payload.get(key) for key in ("base", "epoch", "global_step", "image_size", "model_arch")}
    metadata.update(path=str(path.resolve()), weights=source)
    return model.eval().requires_grad_(False).to(device), metadata
