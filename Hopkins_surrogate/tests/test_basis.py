import torch

from hopkins_surrogate.basis import (build_correction_basis,
                                     cubic_bspline_matrix,
                                     signed_bin_index)


def _subgrid(n_fft=800, kc=10):
    half = torch.arange(-kc, kc + 1)
    ky, kx = torch.meshgrid(half, half, indexing="ij")
    shape = (2 * kc + 1, 2 * kc + 1)
    return ky.reshape(-1) % n_fft, kx.reshape(-1) % n_fft, shape


def test_signed_bins():
    n_fft, kc = 800, 10
    sub_y, sub_x, _ = _subgrid(n_fft, kc)
    signed_y = signed_bin_index(sub_y, n_fft)
    signed_x = signed_bin_index(sub_x, n_fft)
    assert int(signed_y.min()) == -kc
    assert int(signed_y.max()) == kc
    assert int(signed_x.min()) == -kc
    assert int(signed_x.max()) == kc
    assert int(signed_y[10 * 21 + 10]) == 0
    assert int(signed_x[10 * 21 + 10]) == 0


def test_bspline_partition_of_unity():
    u = torch.linspace(0.0, 1.0, 101)
    a = cubic_bspline_matrix(u, 5)
    assert a.shape == (101, 5)
    assert torch.all(a >= -1e-7)
    assert torch.allclose(a.sum(dim=1), torch.ones(101), atol=1e-5)
    assert torch.allclose(a[0], torch.tensor([1.0, 0, 0, 0, 0]), atol=1e-6)
    assert torch.allclose(a[-1], torch.tensor([0.0, 0, 0, 0, 1]), atol=1e-6)


def test_basis_dc_and_window():
    n_fft, kc = 800, 10
    sub_y, sub_x, shape = _subgrid(n_fft, kc)
    basis = build_correction_basis(sub_y, sub_x, n_fft, shape, n_basis=5)
    nsub = shape[0] * shape[1]
    assert basis.basis_dc.shape == (nsub, 25)
    assert torch.allclose(basis.basis_dc[basis.dc_index],
                          torch.zeros(25), atol=1e-7)
    assert float(basis.window[basis.dc_index]) == 1.0
    corner = int(torch.argmax(basis.signed_k[:, 0].abs()
                              + basis.signed_k[:, 1].abs()))
    assert float(basis.window[corner]) == 0.0
    assert torch.equal(basis.signed_k[basis.dc_index],
                       torch.zeros(2, dtype=torch.int64))
    assert bool((basis.window >= 0).all()) and bool((basis.window <= 1).all())
