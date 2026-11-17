import pytest
import torch

from hopkins_surrogate.checkpoint import (FingerprintMismatchError,
                                          build_model_from_checkpoint,
                                          load_checkpoint, save_checkpoint)
from hopkins_surrogate.config import SurrogateConfig, TrainConfig
from hopkins_surrogate.imaging import CorrectedSocsImaging

SCALE = 0.86


def _model():
    cfg = SurrogateConfig(
        grid_size=96, pixel_size_nm=4.0,
        source_rings=2, source_points_per_ring=8,
        num_orders=4, aerial_scale=SCALE,
        alpha_init=0.01, c_init_std=0.01)
    model = CorrectedSocsImaging(
        cfg.optics(SCALE), "cpu", num_orders=4, basis_size=5,
        alpha_init=0.01, c_init_std=0.01, seed=0)
    return model, cfg


def test_checkpoint_roundtrip(tmp_path):
    model, cfg = _model()
    mask = torch.zeros((96, 96))
    mask[20:40, 20:50] = 1.0
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, None, None, 1, 1, 0.0,
                    TrainConfig(), cfg, {"s": SCALE}, "split-hash")
    ckpt = load_checkpoint(path)
    loaded, _ = build_model_from_checkpoint(ckpt, "cpu")
    with torch.no_grad():
        out0 = model(mask, use_correction=True)
        out1 = loaded(mask, use_correction=True)
    assert torch.equal(out0, out1)


def test_fingerprint_mismatch(tmp_path):
    model, cfg = _model()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, None, None, 1, 1, 0.0,
                    TrainConfig(), cfg, {"s": SCALE}, "split-hash")
    ckpt = load_checkpoint(path)
    ckpt["fingerprint"] = dict(ckpt["fingerprint"])
    ckpt["fingerprint"]["kernels_sha256"] = "bad"
    with pytest.raises(FingerprintMismatchError):
        build_model_from_checkpoint(ckpt, "cpu", strict=True)
