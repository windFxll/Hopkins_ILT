import pytest
import torch

from hopkins_surrogate import DEFAULT_CHECKPOINT, HopkinsSurrogate


@pytest.fixture(scope="module")
def model():
    return HopkinsSurrogate.from_pretrained(device="cpu")


def _line_space_mask() -> torch.Tensor:
    # 100 nm lines at 400 nm pitch (25% coverage), periodic.
    mask = torch.zeros(800, 800)
    for start in range(0, 800, 400):
        mask[:, start:start + 100] = 1.0
    return mask


def test_metadata(model):
    assert DEFAULT_CHECKPOINT.is_file()
    assert model.grid_size == 800
    assert model.pixel_size_nm == 1.0
    assert model.threshold == pytest.approx(0.346585)
    assert model.tau == pytest.approx(model.threshold)
    assert model.num_orders == 8
    assert model.config.resist_threshold == pytest.approx(0.346585)
    assert not any(p.requires_grad for p in model.parameters())


def test_forward_and_predict(model):
    mask = _line_space_mask()
    with torch.no_grad():
        aerial = model(mask)
    assert aerial.shape == (800, 800)
    assert torch.isfinite(aerial).all()
    assert 0.0 < float(aerial.mean()) < 1.0

    out = model.predict(mask)
    assert set(out) == {"aerial", "clear", "printed", "threshold"}
    assert out["clear"].shape == (800, 800)
    assert out["clear"].dtype == torch.bool
    assert out["printed"].dtype == torch.bool
    assert out["threshold"] == pytest.approx(0.346585)
    # feature lines are dark; clear region is the majority
    assert float(out["clear"].float().mean()) > 0.5
    assert bool(out["printed"][0, 0])


def test_batch_and_numpy(model):
    mask = _line_space_mask()
    batch = torch.stack([mask, mask])
    with torch.no_grad():
        aerial = model(batch)
    assert aerial.shape == (2, 800, 800)
    with torch.no_grad():
        aerial_np = model(mask.numpy())
    assert aerial_np.shape == (800, 800)


def test_gradient_flows_to_mask(model):
    mask = _line_space_mask().requires_grad_(True)
    aerial = model(mask)
    loss = aerial.mean()
    loss.backward()
    assert mask.grad is not None
    assert torch.isfinite(mask.grad).all()
    assert float(mask.grad.abs().sum()) > 0.0


def test_rejects_wrong_shape(model):
    with pytest.raises(ValueError, match="spatial shape"):
        model(torch.zeros(256, 256))
    with pytest.raises(ValueError, match="\\[H, W\\]"):
        model(torch.zeros(2, 2, 800, 800))
