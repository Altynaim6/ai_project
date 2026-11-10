from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/hai"
ARTIFACTS = ROOT / "artifacts/hai/preprocessing"

EXCLUDED = ["time", "attack", "attack_P1", "attack_P2", "attack_P3"]


def get_features(frame, columns):
    values = frame[columns].apply(pd.to_numeric, errors="raise")

    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Features contain missing or infinite values")

    return values


def main():
    train = pd.read_csv(DATA / "train.csv", sep=";")
    validation = pd.read_csv(DATA / "validation.csv", sep=";")

    if list(train.columns) != list(validation.columns):
        raise ValueError("Train and validation schemas differ")

    for frame in (train, validation):
        if not frame["attack"].eq(0).all():
            raise ValueError("Expected normal-only data")

    columns = [name for name in train.columns if name not in EXCLUDED]

    train_features = get_features(train, columns)
    validation_features = get_features(validation, columns)

    scaler = StandardScaler()

    # Fit выполняется исключительно на train.
    train_scaled = scaler.fit_transform(train_features)
    validation_scaled = scaler.transform(validation_features)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "scaler": scaler,
            "feature_columns": columns,
            "excluded_columns": EXCLUDED,
            "dataset": "HAI 20.07",
        },
        ARTIFACTS / "scaler.joblib",
    )

    for name, frame, values in (
        ("train", train, train_scaled),
        ("validation", validation, validation_scaled),
    ):
        np.savez_compressed(
            DATA / f"{name}_scaled.npz",
            features=values.astype(np.float32),
            timestamps=frame["time"].to_numpy(dtype=str),
            labels=frame["attack"].to_numpy(dtype=np.int64),
            feature_columns=np.array(columns, dtype=str),
        )

    print("Train shape:", train_scaled.shape)
    print("Validation shape:", validation_scaled.shape)
    print("Constant train features:",
          int((train_features.nunique() <= 1).sum()))
    print("Scaler saved:", ARTIFACTS / "scaler.joblib")
    print("Scaling completed")


if __name__ == "__main__":
    main()