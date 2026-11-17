"""Short ILT run using the final surrogate as the imaging model.

This is a smoke example, not a tuned production ILT run.  It shows the only
integration point needed: ``PixelILT`` accepts any callable imaging model with
a ``.config.resist_threshold`` attribute, and the surrogate is differentiable
with respect to the mask.

    pip install -e Hopkins -e ILT -e Hopkins_surrogate
    python Hopkins_surrogate/examples/ilt_with_surrogate.py
"""

from dataclasses import replace
from pathlib import Path

import torch

from hopkins.resist import hard_resist
from hopkins_surrogate import HopkinsSurrogate
from ilt.config import OptimizationConfig
from ilt.layouts import realistic_interconnect
from ilt.optimization import PixelILT
from ilt.utils.metrics import binary_iou
from _viz import save_ilt

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = HopkinsSurrogate.from_pretrained(device=device)

    # The surrogate is fixed to 800 x 800 @ 1 nm/px.  ILT's default demo
    # profile is 256 x 256 @ 4 nm, so scale the filter/edge parameters to
    # nanometres (same values as ILT's `iccad-native` profile).
    opt_cfg = replace(
        OptimizationConfig(),
        iterations=100,
        log_every=10,
        early_stop=False,
        mask_filter_sigma_px=5.0,
        border_margin_px=40,
        boundary_radius_px=8,
        tv_weight=2.0e-3,
    )
    target = torch.as_tensor(
        realistic_interconnect(model.grid_size, model.pixel_size_nm),
        dtype=torch.float32, device=model.device)

    print(f"target: {tuple(target.shape)} | 1 = feature | "
          f"threshold={model.threshold}")
    with torch.no_grad():
        initial_printed = hard_resist(model(target), model.threshold)
    result = PixelILT(model, opt_cfg).optimize(target)

    printed = hard_resist(result.optimized_aerial, model.threshold)
    iou = binary_iou(printed, target)
    figure = ROOT / "output" / "examples" / "ilt_with_surrogate.png"
    save_ilt(figure, target, initial_printed, result, model.pixel_size_nm)
    print(f"finished in {result.elapsed_seconds:.2f} s | "
          f"steps={result.steps_completed} | IoU={iou:.4f}")
    print(f"figure: {figure}")


if __name__ == "__main__":
    main()
