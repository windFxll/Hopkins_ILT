import numpy as np
import torch

from hopkins_surrogate.losses import (band_loss, contour_weight, dice_loss,
                                      relative_loss)
from hopkins_surrogate.metrics import (binary_dice, binary_iou,
                                       contour_mean_distance_nm,
                                       evaluate_sample)

TAU = 0.346585


def test_contour_weight():
    t = torch.tensor([TAU, TAU + 10.0])
    w = contour_weight(t, TAU, 0.08)
    assert torch.allclose(w[0], torch.tensor(1.0), atol=1e-6)
    assert float(w[1]) < 0.101


def test_band_and_relative_loss_zero():
    x = torch.rand(32, 32)
    assert float(band_loss(x, x, TAU, 0.08)) < 1e-10
    assert float(relative_loss(x, x)) < 1e-10


def test_dice_perfect():
    pred = torch.full((16, 16), 1.0)
    binary = torch.ones((16, 16))
    assert float(dice_loss(pred, binary, TAU, 0.03)) < 1e-4
    pred0 = torch.zeros((16, 16))
    binary0 = torch.zeros((16, 16))
    assert float(dice_loss(pred0, binary0, TAU, 0.03)) < 1e-2


def test_iou_dice_simple():
    a = np.zeros((8, 8), dtype=bool)
    b = np.zeros((8, 8), dtype=bool)
    a[2:4, 2:4] = True
    b[2:4, 2:4] = True
    assert binary_iou(a, b) == 1.0
    assert binary_dice(a, b) == 1.0
    b[4:6, 4:6] = True
    assert 0.0 < binary_iou(a, b) < 1.0


def test_contour_shift():
    a = np.zeros((64, 64), dtype=bool)
    b = np.zeros((64, 64), dtype=bool)
    a[10:20, 10:20] = True
    b[12:22, 10:20] = True
    cd = contour_mean_distance_nm(a, b, pixel_size_nm=1.0)
    assert 0.5 < cd < 2.5


def test_evaluate_sample_identity():
    target = np.full((32, 32), 0.9, dtype=np.float32)
    target[8:24, 8:24] = 0.1
    binary = target > TAU
    out = evaluate_sample(target, target, binary)
    assert out["iou_clear"] == 1.0
    assert out["dice_clear"] == 1.0
    assert out["band_mse"] < 1e-10
    assert out["contour_nm"] == 0.0
