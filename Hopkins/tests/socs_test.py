import argparse
from dataclasses import replace
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hopkins import AbbeImaging, OpticsConfig, SocsImaging, make_imaging
from hopkins.pupil import build_shifted_pupils


def deterministic_mask(n, device):
    mask = torch.zeros((n, n), device=device)
    mask[36:60, 28:68] = 1.0
    mask[20:30, 10:80] = 1.0
    mask[52:58, 70:88] = 1.0
    return mask


def rms(a, b):
    return float((a - b).square().mean().sqrt())


def compare_full_rank(cfg, device, label):
    mask = deterministic_mask(cfg.grid_size, device)
    abbe = AbbeImaging(cfg, device)
    socs = SocsImaging(cfg, device, num_orders=None, eigh_mode="low_rank")
    with torch.no_grad():
        a = abbe(mask)
        s = socs(mask)
    err_rms = rms(a, s)
    err_max = float((a - s).abs().max())
    assert err_rms < 1e-5, f"{label}: RMS {err_rms:.3e} >= 1e-5"
    assert err_max < 1e-4, f"{label}: max {err_max:.3e} >= 1e-4"
    print(f"PASS full-rank {label}: rank={socs.rank} "
          f"rms={err_rms:.2e} max={err_max:.2e}")
    return abbe, socs, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--full", action="store_true",
                        help="also check the default 256 configuration")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    base = OpticsConfig(grid_size=96, pixel_size_nm=4.0,
                        source_rings=2, source_points_per_ring=8,
                        source_batch_size=4, boundary_mode="isolated")
    compare_full_rank(base, device, "isolated")
    compare_full_rank(replace(base, boundary_mode="periodic"), device,
                      "periodic")
    compare_full_rank(replace(base, boundary_mode="periodic", defocus_nm=50.0),
                      device, "periodic+defocus")
    compare_full_rank(replace(base, defocus_nm=40.0,
                              resist_blur_sigma_nm=2.0),
                      device, "isolated+defocus+blur")

    abbe, socs, mask = compare_full_rank(base, device, "base")

    # Gradient agreement.
    m1 = mask.clone().requires_grad_(True)
    m2 = mask.clone().requires_grad_(True)
    abbe(m1).square().mean().backward()
    socs(m2).square().mean().backward()
    grad_err = float((m1.grad - m2.grad).abs().max() / m1.grad.abs().max())
    assert grad_err < 1e-4, f"gradient relative error {grad_err:.3e}"
    print(f"PASS gradient relative error {grad_err:.2e}")

    # Subgrid must cover every non-zero pupil entry.
    pupils, _, _ = build_shifted_pupils(base, device)
    support = pupils.any(dim=0).detach().cpu()
    covered = torch.zeros(support.shape, dtype=torch.bool)
    covered[socs.sub_y.cpu(), socs.sub_x.cpu()] = True
    missing = int((support & ~covered).sum())
    assert missing == 0, f"{missing} non-zero pupil entries outside subgrid"
    print(f"PASS subgrid coverage: {socs.subgrid_shape}")

    # Empty mask is 1 for every truncation order.
    with torch.no_grad():
        empty_mask = torch.zeros((base.grid_size, base.grid_size),
                                 device=device)
        for k in (1, 2, 4, 8, socs.max_orders):
            empty = socs(empty_mask, num_orders=k)
            assert float((empty - 1.0).abs().max()) < 1e-5, \
                f"empty mask not 1 at K={k}"
    print("PASS empty mask normalization")

    # Clear-field intensity matches Abbe.
    clear_err = abs(float(socs.clear_intensity)
                    - float(abbe.clear_intensity))
    assert clear_err < 1e-5, f"clear intensity differs by {clear_err:.3e}"
    print(f"PASS clear intensity error {clear_err:.2e}")

    # Energy monotonic and truncation error decreases.
    with torch.no_grad():
        full = socs(mask)
    energies = [socs.energy_fraction(k) for k in (1, 2, 4, 8, 16)]
    assert all(b >= a for a, b in zip(energies, energies[1:])), energies
    assert abs(socs.energy_fraction(socs.max_orders) - 1.0) < 1e-5
    errors = []
    for k in (1, 2, 4, 8, 16):
        with torch.no_grad():
            errors.append(rms(socs(mask, num_orders=k), full))
    assert all(b <= a + 1e-6 for a, b in zip(errors, errors[1:])), errors
    print(f"PASS truncation energy={['%.4f' % e for e in energies]} "
          f"rms={['%.2e' % e for e in errors]}")

    # Explicit TCC path agrees with the low-rank path.
    explicit = SocsImaging(base, device, num_orders=None, eigh_mode="explicit")
    with torch.no_grad():
        se = explicit(mask)
        sl = socs(mask)
    path_err = rms(se, sl)
    assert path_err < 1e-5, f"explicit vs low_rank RMS {path_err:.3e}"
    print(f"PASS explicit vs low_rank rms={path_err:.2e}")

    # Interface and factory.
    model = make_imaging(base, device, method="socs", num_orders=4)
    assert isinstance(model, SocsImaging)
    assert model.config is base
    assert tuple(model(mask).shape) == (base.grid_size, base.grid_size)
    assert isinstance(make_imaging(base, device, method="abbe"), AbbeImaging)
    try:
        make_imaging(base, device, method="unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown method should raise ValueError")
    print("PASS factory/interface")

    if args.full:
        full_cfg = OpticsConfig()
        compare_full_rank(full_cfg, device, "default-256")

    print("ALL SOCS TESTS PASS")


if __name__ == "__main__":
    main()
