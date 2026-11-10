"""Real-data forward/backward and checkpoint restoration smoke check."""
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT, load_config
from src.mixed_forecast_loss import mixed_forecast_scores
from src.mixed_window_dataset import MixedForecastWindowDataset
from src.models.transformer_mixed import MixedForecastTransformer


def main():
    config = load_config()
    torch.manual_seed(config["experiment"]["seeds"][0])
    length = config["experiment"]["sequence_length"]
    datasets = {
        split: MixedForecastWindowDataset(
            PROJECT_ROOT / f"data/processed/hai_mixed/{split}.npz", length
        ) for split in ("train", "validation")
    }
    train, validation = datasets.values()
    if (train.numeric_columns != validation.numeric_columns
            or train.binary_columns != validation.binary_columns
            or train.schema_sha256 != validation.schema_sha256):
        raise ValueError("Split schemas differ")
    if train.timestamps[-1] >= validation.timestamps[0]:
        raise ValueError("Split timestamps overlap")

    model = MixedForecastTransformer(sequence_length=length)
    for split, dataset in datasets.items():
        history, numeric, binary = next(iter(DataLoader(dataset, batch_size=32)))
        model.zero_grad(set_to_none=True)
        prediction, logits = model(history)
        loss = mixed_forecast_scores(prediction, logits, numeric, binary).mean()
        loss.backward()
        for head in (model.numeric_head, model.binary_head, model.input_projection):
            assert head.weight.grad is not None
            assert torch.isfinite(head.weight.grad).all()
            assert head.weight.grad.abs().sum() > 0
        print(f"{split}: {len(dataset)} windows; input={tuple(history.shape)}; "
              f"numeric={tuple(prediction.shape)}; logits={tuple(logits.shape)}; loss={loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        expected = model(history)
    # Temporary checkpoint, not a trained artifact.
    with tempfile.TemporaryDirectory(prefix=".mixed-check-", dir=PROJECT_ROOT) as directory:
        path = Path(directory) / "check.pt"
        torch.save({"model_config": model.model_config,
                    "model_state_dict": model.state_dict(),
                    "numeric_columns": train.numeric_columns,
                    "binary_columns": train.binary_columns,
                    "schema_sha256": train.schema_sha256}, path)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        restored = MixedForecastTransformer(**saved["model_config"])
        restored.load_state_dict(saved["model_state_dict"])
        restored.eval()
        with torch.no_grad():
            actual = restored(history)
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, rtol=0, atol=1e-5)
    print("Both heads/backward and save/load: OK (model not trained)")


if __name__ == "__main__":
    main()
