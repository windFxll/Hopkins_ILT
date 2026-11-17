from __future__ import annotations

import hashlib
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from hopkins.config import OpticsConfig

from .config import SurrogateConfig, TrainConfig
from .imaging import CorrectedSocsImaging


class FingerprintMismatchError(RuntimeError):
    pass


def _sha256_tensor(t: torch.Tensor) -> str:
    arr = t.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def kernel_fingerprint(socs, num_orders: int) -> dict:
    q = max(1, min(int(num_orders), int(socs.max_orders)))
    return {
        "num_orders": q,
        "rank": int(socs.rank),
        "subgrid_shape": [int(v) for v in socs.subgrid_shape],
        "kernels_sha256": _sha256_tensor(socs.kernels[:q]),
        "eigenvalues_sha256": _sha256_tensor(socs.eigenvalues[:q]),
        "subgrid_sha256": _sha256_tensor(
            torch.stack([socs.sub_y, socs.sub_x])),
    }


def save_checkpoint(path: str | Path, wrapper: CorrectedSocsImaging,
                    optimizer, scheduler, epoch: int, global_step: int,
                    best_val: float, train_cfg: TrainConfig,
                    surrogate_cfg: SurrogateConfig,
                    calibration: dict, split_hash: str,
                    rng_states: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_state": wrapper.correction.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val": float(best_val),
        "train_config": train_cfg.to_dict(),
        "surrogate_config": surrogate_cfg.to_dict(),
        "optics": asdict(wrapper.config),
        "correction_enabled": bool(wrapper.correction_enabled),
        "calibration": calibration,
        "split_hash": split_hash,
        "fingerprint": kernel_fingerprint(wrapper.base, wrapper.num_orders),
        "rng_states": rng_states,
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    return torch.load(Path(path), map_location=map_location,
                      weights_only=False)


def build_model_from_checkpoint(ckpt: dict, device,
                                strict: bool = True
                                ) -> tuple[CorrectedSocsImaging, dict]:
    optics = OpticsConfig(**ckpt["optics"])
    sc = SurrogateConfig.from_dict(ckpt.get("surrogate_config", {}))
    wrapper = CorrectedSocsImaging(
        optics, device,
        num_orders=int(ckpt.get("fingerprint", {}).get(
            "num_orders", sc.num_orders)),
        basis_size=int(sc.basis_size),
        alpha_max=float(sc.alpha_max),
        alpha_init=float(sc.alpha_init),
        c_init_std=float(sc.c_init_std),
        correction_enabled=bool(ckpt.get("correction_enabled", True)),
        window_inner=float(sc.window_inner),
        window_outer=float(sc.window_outer),
    )
    wrapper.correction.load_state_dict(ckpt["model_state"])

    expected = ckpt["fingerprint"]
    actual = kernel_fingerprint(wrapper.base, wrapper.num_orders)
    if actual != expected:
        message = (
            "SOCS eigenbasis fingerprint mismatch; retrain under the same "
            "OpticsConfig/TCC construction.\n"
            f"expected={expected}\nactual={actual}")
        if strict:
            raise FingerprintMismatchError(message)
        print("WARNING:", message)

    calib = ckpt.get("calibration") or {}
    if "s" in calib:
        s = float(calib["s"])
        if abs(float(optics.aerial_scale) - s) > 1e-9:
            message = (f"aerial_scale {optics.aerial_scale} != "
                       f"calibration s {s}")
            if strict:
                raise FingerprintMismatchError(message)
            print("WARNING:", message)
    return wrapper, ckpt


def collect_rng_states() -> dict:
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states
