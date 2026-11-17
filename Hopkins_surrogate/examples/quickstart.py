"""Minimal inference example for the final K=8 surrogate model.

Run after installing the packages:

    pip install -e Hopkins -e Hopkins_surrogate
    python Hopkins_surrogate/examples/quickstart.py
"""

import time
from pathlib import Path

import torch

from hopkins_surrogate import HopkinsSurrogate
from _viz import save_inference

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    model = HopkinsSurrogate.from_pretrained(device=device)
    load_s = time.perf_counter() - t0

    # Periodic 100 nm lines / 400 nm pitch on the 800 x 800 @ 1 nm grid.
    mask = torch.zeros(model.grid_size, model.grid_size, device=model.device)
    for start in range(0, model.grid_size, 400):
        mask[:, start:start + 100] = 1.0

    # Warm up CUDA kernels before timing.
    with torch.no_grad():
        model(mask)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.predict(mask)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    infer_ms = (time.perf_counter() - t0) * 1000.0

    aerial = out["aerial"]
    print(repr(model))
    print(f"load {load_s:.2f} s | forward {infer_ms:.1f} ms | "
          f"device {model.device}")
    print(f"aerial: shape={tuple(aerial.shape)} "
          f"min={aerial.min():.4f} max={aerial.max():.4f}")
    print(f"clear fraction: {out['clear'].float().mean():.4f}")
    print(f"printed fraction: {out['printed'].float().mean():.4f}")
    figure = ROOT / "output" / "examples" / "quickstart.png"
    save_inference(figure, mask, out)
    print(f"figure: {figure}")

    # ILT needs gradients with respect to the mask.  `predict` is
    # inference-only; use the callable forward for optimization.
    trainable_mask = mask.clone().requires_grad_(True)
    loss = model(trainable_mask).mean()
    loss.backward()
    print(f"mask gradient: {tuple(trainable_mask.grad.shape)} "
          f"|grad| mean={trainable_mask.grad.abs().mean():.3e}")


if __name__ == "__main__":
    main()
