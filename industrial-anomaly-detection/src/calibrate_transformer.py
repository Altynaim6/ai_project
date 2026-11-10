import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT
from src.models.transformer import ForecastTransformer
from src.window_dataset import ForecastWindowDataset


def main():
    model_dir = PROJECT_ROOT / "artifacts/hai/transformer_smoke"

    checkpoint = torch.load(
        model_dir / "best_model.pt",
        map_location="cpu",
        weights_only=True,
    )

    model = ForecastTransformer(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sequence_length = checkpoint["model_config"]["sequence_length"]
    dataset = ForecastWindowDataset(
        PROJECT_ROOT / "data/processed/hai/validation_scaled.npz",
        sequence_length=sequence_length,
    )

    if dataset.feature_columns.tolist() != checkpoint["feature_columns"]:
        raise ValueError("Feature order differs from the model")

    if not np.all(dataset.labels == 0):
        raise ValueError("Calibration requires normal-only validation")

    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    scores = []

    with torch.no_grad():
        for history, target in loader:
            prediction = model(history)
            batch_scores = (prediction - target).square().mean(dim=1)
            scores.append(batch_scores.numpy())

    scores = np.concatenate(scores)

    if not np.isfinite(scores).all():
        raise ValueError("Non-finite anomaly scores")

    threshold = float(np.quantile(scores, 0.99))
    predictions = (scores > threshold).astype(np.int64)

    output_dir = PROJECT_ROOT / "results/hai/transformer_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "timestamp": dataset.timestamps[sequence_length:],
        "label": dataset.labels[sequence_length:],
        "score": scores,
        "prediction": predictions,
    }).to_csv(output_dir / "validation_scores.csv", index=False)

    calibration = {
        "model_file": "best_model.pt",
        "run_type": "smoke_test",
        "method": "normal_validation_quantile",
        "quantile": 0.99,
        "decision_rule": "score > threshold",
        "threshold": threshold,
        "validation_windows": len(scores),
        "validation_fpr": float(predictions.mean()),
        "note": "Exploratory threshold; not a final experiment result",
    }

    (model_dir / "threshold.json").write_text(
        json.dumps(calibration, indent=2),
        encoding="utf-8",
    )

    print("Validation windows:", len(scores))
    print("Mean score:", float(scores.mean()))
    print("Threshold:", threshold)
    print("Validation FPR:", float(predictions.mean()))
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()