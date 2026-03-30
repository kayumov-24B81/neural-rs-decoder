import torch

from src.model import PositionPredictor


def test_forward_pass():
    """Model forward pass runs without error."""
    model = PositionPredictor()
    model.eval()
    x = torch.randn(1, 511)
    out = model(x)
    assert out.shape == (1, 255)


def test_output_range_after_sigmoid():
    """Output values in [0, 1] after sigmoid."""
    model = PositionPredictor()
    x = torch.randn(4, 511)
    probs = torch.sigmoid(model(x))
    assert probs.min() >= 0 and probs.max() <= 1
