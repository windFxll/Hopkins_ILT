import torch

from hopkins.config import OpticsConfig
from hopkins_surrogate.imaging import CorrectedSocsImaging
from hopkins_surrogate.losses import band_loss

SCALE = 0.86
TAU = 0.346585


def _config():
    return OpticsConfig(
        grid_size=96, pixel_size_nm=4.0,
        source_rings=2, source_points_per_ring=8,
        boundary_mode="periodic", aerial_scale=SCALE)


def test_smoke_overfit():
    torch.manual_seed(0)
    mask = torch.zeros((96, 96))
    mask[25:55, 30:70] = 1.0
    teacher = CorrectedSocsImaging(
        _config(), "cpu", num_orders=4, basis_size=5,
        alpha_max=0.05, alpha_init=0.01, c_init_std=0.02, seed=1)
    with torch.no_grad():
        target = teacher(mask, use_correction=True).detach()

    student = CorrectedSocsImaging(
        _config(), "cpu", num_orders=4, basis_size=5,
        alpha_max=0.05, alpha_init=0.01, c_init_std=0.0, seed=2)
    optimizer = torch.optim.Adam(
        [student.correction.c_real, student.correction.c_imag], lr=0.05)
    initial = None
    loss = None
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        pred = student(mask, use_correction=True)
        loss = band_loss(pred, target, TAU, 0.08 * SCALE)
        if initial is None:
            initial = float(loss.detach())
        loss.backward()
        optimizer.step()
    final = float(loss.detach())
    assert final < initial * 0.9
