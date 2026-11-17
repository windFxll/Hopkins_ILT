import torch

from hopkins.config import OpticsConfig
from hopkins_surrogate.imaging import CorrectedSocsImaging


def _config(scale=0.86):
    return OpticsConfig(
        grid_size=96, pixel_size_nm=4.0,
        source_rings=2, source_points_per_ring=8,
        boundary_mode="periodic", aerial_scale=scale)


def _model(scale=0.86):
    return CorrectedSocsImaging(
        _config(scale), "cpu", num_orders=4, basis_size=5,
        alpha_max=0.05, alpha_init=0.01, c_init_std=0.01, seed=0)


def _mask():
    m = torch.zeros((96, 96))
    m[20:45, 30:65] = 1.0
    m[60:70, 15:80] = 1.0
    return m


def test_off_bit_exact():
    model = _model()
    mask = _mask()
    off = model(mask, use_correction=False)
    base = model.base(mask)
    assert torch.equal(off, base)


def test_on_changes_output_and_grad():
    model = _model()
    mask = _mask()
    off = model(mask, use_correction=False)
    on = model(mask, use_correction=True)
    assert not torch.equal(on, off)
    on.square().mean().backward()
    grad = model.correction.c_real.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0


def test_empty_mask_scaled():
    model = _model()
    empty = torch.zeros((96, 96))
    out = model(empty)
    assert torch.allclose(out, torch.full_like(out, 0.86), atol=1e-5)


def test_batch_matches_loop():
    model = _model()
    mask = _mask()
    batch = torch.stack([mask, mask.flip(0)], dim=0)
    out = model(batch)
    ref = torch.stack([model(mask), model(mask.flip(0))], dim=0)
    assert torch.equal(out, ref)


def test_disable_correction():
    model = _model()
    model.set_correction_enabled(False)
    mask = _mask()
    assert torch.equal(model(mask), model.base(mask))


def test_projection_bounds_relative_norm():
    model = _model()
    with torch.no_grad():
        model.correction.c_real.normal_(0.0, 1.0)
        model.correction.c_imag.normal_(0.0, 1.0)
    model.correction.project_relative_norm(
        model.base.kernels, mean_max=0.05, mode_max=0.10)
    rel = model.correction.relative_norm(model.base.kernels)
    assert float(rel.mean()) <= 0.05 + 1e-6
    assert float(rel.max()) <= 0.10 + 1e-6
