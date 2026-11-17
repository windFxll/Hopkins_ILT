from dataclasses import replace

import torch

from hopkins import AbbeImaging, SocsImaging


def _mask(device):
    m = torch.zeros((96, 96), device=device)
    m[20:45, 30:65] = 1.0
    m[60:70, 15:80] = 1.0
    return m


def test_socs_kernel_override_bit_exact(small_config, device):
    socs = SocsImaging(small_config, device, num_orders=4)
    mask = _mask(device)
    out = socs(mask)
    out_override = socs(mask, kernels=socs.kernels)
    assert torch.equal(out, out_override)


def test_socs_scale_exact(small_config, device):
    cfg_s = replace(small_config, aerial_scale=0.86)
    socs0 = SocsImaging(small_config, device, num_orders=4)
    socs_s = SocsImaging(cfg_s, device, num_orders=4)
    mask = _mask(device)
    out0 = socs0(mask)
    out_s = socs_s(mask)
    assert torch.equal(out_s, out0 * 0.86)


def test_empty_mask_scale(small_config, device):
    cfg_s = replace(small_config, aerial_scale=0.86)
    socs_s = SocsImaging(cfg_s, device, num_orders=4)
    empty = torch.zeros((96, 96), device=device)
    out = socs_s(empty)
    assert torch.allclose(out, torch.full_like(out, 0.86), atol=1e-5)


def test_abbe_scale_exact(small_config, device):
    cfg_s = replace(small_config, aerial_scale=0.86)
    abbe0 = AbbeImaging(small_config, device)
    abbe_s = AbbeImaging(cfg_s, device)
    mask = _mask(device)
    assert torch.equal(abbe_s(mask), abbe0(mask) * 0.86)


def test_kernel_gradient(small_config, device):
    socs = SocsImaging(small_config, device, num_orders=4)
    mask = _mask(device)
    kernels = socs.kernels[:4].clone().detach().requires_grad_(True)
    out = socs(mask, kernels=kernels)
    out.square().mean().backward()
    assert kernels.grad is not None
    assert torch.isfinite(kernels.grad).all()
    assert float(kernels.grad.abs().sum()) > 0.0
