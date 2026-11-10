import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT
from src.models.transformer import ForecastTransformer
from src.window_dataset import ForecastWindowDataset


def main():
    model_dir = PROJECT_ROOT / "artifacts/hai/transformer_smoke"
    output_dir = PROJECT_ROOT / "results/hai/transformer_smoke"

    checkpoint = torch.load(
        model_dir / "best_model.pt",
        map_location="cpu",
        weights_only=True,
    )
    preprocessing = joblib.load(
        PROJECT_ROOT / "artifacts/hai/preprocessing/scaler.joblib"
    )

    model = ForecastTransformer(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    length = checkpoint["model_config"]["sequence_length"]
    dataset = ForecastWindowDataset(
        PROJECT_ROOT / "data/processed/hai/validation_scaled.npz",
        sequence_length=length,
    )
    columns = dataset.feature_columns.tolist()

    if columns != checkpoint["feature_columns"]:
        raise ValueError("Model feature order differs")
    if columns != preprocessing["feature_columns"]:
        raise ValueError("Scaler feature order differs")

    scaler = preprocessing["scaler"]
    constant_features = scaler.var_ == 0

    scores_file = pd.read_csv(
        output_dir / "validation_scores.csv"
    )
    if len(scores_file) != len(dataset):
        raise ValueError("Saved score count differs")

    top_indices = (
        scores_file["score"].nlargest(10).index.to_numpy()
    )
    selected = set(top_indices.tolist())

    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    total_errors = np.zeros(len(columns), dtype=np.float64)
    peak_rows = []
    offset = 0

    with torch.no_grad():
        for history, target in loader:
            prediction = model(history)
            errors = (prediction - target).square().numpy()
            total_errors += errors.sum(axis=0, dtype=np.float64)

            for local_index in range(len(history)):
                window_index = offset + local_index
                if window_index not in selected:
                    continue

                predicted_raw = scaler.inverse_transform(
                    prediction[local_index:local_index + 1].numpy()
                )[0]
                actual_raw = scaler.inverse_transform(
                    target[local_index:local_index + 1].numpy()
                )[0]

                feature_errors = errors[local_index]
                largest = np.argsort(feature_errors)[-5:][::-1]

                for rank, feature_index in enumerate(largest, start=1):
                    peak_rows.append({
                        "timestamp": dataset.timestamps[
                            window_index + length
                        ],
                        "window_index": window_index,
                        "score": float(feature_errors.mean()),
                        "rank": rank,
                        "feature": columns[feature_index],
                        "squared_error_scaled":
                            float(feature_errors[feature_index]),
                        "actual_raw": float(actual_raw[feature_index]),
                        "predicted_raw":
                            float(predicted_raw[feature_index]),
                        "constant_on_train":
                            bool(constant_features[feature_index]),
                    })

            offset += len(history)

    summary = pd.DataFrame({
        "feature": columns,
        "mean_squared_error": total_errors / len(dataset),
        "constant_on_train": constant_features,
    }).sort_values("mean_squared_error", ascending=False)

    peaks = pd.DataFrame(peak_rows).sort_values(
        ["score", "rank"],
        ascending=[False, True],
    )

    summary.to_csv(output_dir / "feature_errors.csv", index=False)
    peaks.to_csv(output_dir / "peak_errors.csv", index=False)

    print("\nTop 10 channels by validation MSE:")
    print(summary.head(10).to_string(index=False))

    print("\nTop 3 channels at the largest spike:")
    print(peaks.head(3).to_string(index=False))

    print("\nSaved: feature_errors.csv, peak_errors.csv")


if __name__ == "__main__":
    main()