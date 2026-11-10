"""HAI mixed next-state forecast loss. Binary outputs are raw two-class logits."""
import torch
from torch.nn import functional as F


def mixed_forecast_scores(numeric_prediction, binary_logits,
                          numeric_target, binary_target):
    """Return one score per observation: mean MSE + mean categorical CE.

    numeric tensors: [batch, 55], standardized with a numeric-only train scaler.
    binary logits: [batch, 4, 2]; targets: [batch, 4], unscaled 0/1.
    This function expects complete, finite data (as verified for this HAI split).
    """
    if numeric_prediction.shape != numeric_target.shape:
        raise ValueError("Numeric shapes differ")
    if numeric_target.ndim != 2 or numeric_target.shape[1] == 0:
        raise ValueError("Expected numeric targets [batch, channels]")
    if binary_target.ndim != 2 or binary_logits.shape != (*binary_target.shape, 2):
        raise ValueError("Expected binary logits [batch, channels, 2]")
    if binary_target.shape[0] != numeric_target.shape[0]:
        raise ValueError("Batch sizes differ")
    if not torch.all((binary_target == 0) | (binary_target == 1)):
        raise ValueError("Binary targets must be 0 or 1")
    for tensor in (numeric_prediction, numeric_target, binary_logits):
        if not torch.isfinite(tensor).all():
            raise ValueError("Non-finite prediction or target")
    numeric_error = (numeric_prediction - numeric_target).square().mean(dim=1)
    binary_error = F.cross_entropy(
        binary_logits.transpose(1, 2), binary_target.long(), reduction="none"
    ).mean(dim=1)
    return numeric_error + binary_error


if __name__ == "__main__":
    # MSE=1; uniform two-class probabilities give CE=log(2).
    prediction = torch.ones(2, 55, requires_grad=True)
    logits = torch.zeros(2, 4, 2, requires_grad=True)
    scores = mixed_forecast_scores(
        prediction, logits, torch.zeros(2, 55), torch.zeros(2, 4)
    )
    expected = torch.full((2,), 1 + torch.log(torch.tensor(2.)).item())
    torch.testing.assert_close(scores, expected)
    scores.mean().backward()
    assert prediction.grad.abs().sum() > 0
    assert logits.grad.abs().sum() > 0
    print("Mixed loss reference value and gradients: OK")
