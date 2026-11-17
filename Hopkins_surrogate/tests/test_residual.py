import torch

from hopkins.config import OpticsConfig
from hopkins_surrogate.imaging import CorrectedSocsImaging
from hopkins_surrogate.losses import band_loss
from hopkins_surrogate.residual import CorrectedResidualImaging, ResidualCNN


def _config(scale=0.86):
    return OpticsConfig(
        grid_size=96, pixel_size_nm=4.0,
        source_rings=2, source_points_per_ring=8,
        boundary_mode="periodic", aerial_scale=scale)


def _kernel():
    return CorrectedSocsImaging(
        _config(), "cpu", num_orders=4, basis_size=5,
        alpha_max=0.05, alpha_init=0.01, c_init_std=0.0, seed=0)


def _mask():
    m = torch.zeros((96, 96))
    m[20:45, 30:65] = 1.0
    m[60:70, 15:80] = 1.0
    return m


def test_zero_init_residual_and_hybrid_equivalence():
    kernel = _kernel()
    residual = ResidualCNN(hidden=8, delta_max=0.1, downscale=4)
    mask = _mask()
    with torch.no_grad():
        delta = residual(mask, torch.ones_like(mask))
        assert torch.equal(delta, torch.zeros_like(delta))
        hybrid = CorrectedResidualImaging(
            kernel, residual, freeze_kernel=True, detach_socs=True)
        assert torch.equal(
            hybrid(mask, use_residual=True),
            kernel(mask, use_correction=True))


def test_residual_bounded():
    residual = ResidualCNN(hidden=8, delta_max=0.07, downscale=4)
    mask = torch.rand((96, 96))
    intensity = torch.rand((96, 96)) + 0.5
    out = residual(mask, intensity)
    assert out.shape == mask.shape
    assert float(out.abs().max()) <= 0.07 + 1e-6


def test_residual_gradient():
    kernel = _kernel()
    residual = ResidualCNN(hidden=8, delta_max=0.1, downscale=4)
    hybrid = CorrectedResidualImaging(
        kernel, residual, freeze_kernel=True, detach_socs=True)
    mask = _mask()
    with torch.no_grad():
        target = kernel(mask, use_correction=True) + 0.03
    pred = hybrid(mask)
    loss = band_loss(pred, target, tau=0.346585, width=0.08 * 0.86)
    loss.backward()
    assert residual.net[-1].weight.grad is not None
    assert float(residual.net[-1].weight.grad.abs().sum()) > 0.0


def test_kernel_params_frozen():
    kernel = _kernel()
    residual = ResidualCNN(hidden=8, delta_max=0.1, downscale=4)
    hybrid = CorrectedResidualImaging(kernel, residual, freeze_kernel=True)
    assert all(not p.requires_grad for p in hybrid.kernel_model.parameters())
    assert any(p.requires_grad for p in hybrid.residual.parameters())
