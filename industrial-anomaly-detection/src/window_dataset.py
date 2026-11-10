from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ForecastWindowDataset(Dataset):
    def __init__(self, path, sequence_length=30):
        if sequence_length < 1:
            raise ValueError("Sequence length must be positive")

        with np.load(path, allow_pickle=False) as data:
            self.features = np.array(
                data["features"], dtype=np.float32, copy=True
            )
            self.timestamps = data["timestamps"].copy()
            self.labels = data["labels"].copy()
            self.feature_columns = data["feature_columns"].copy()

        if self.features.ndim != 2:
            raise ValueError("Expected features shaped [rows, features]")

        rows = len(self.features)

        if len(self.timestamps) != rows or len(self.labels) != rows:
            raise ValueError("Array lengths differ")

        if rows <= sequence_length:
            raise ValueError("Not enough rows for one forecast window")

        if not np.isfinite(self.features).all():
            raise ValueError("Non-finite feature values")

        timestamps = self.timestamps.astype("datetime64[s]")
        if np.isnat(timestamps).any():
            raise ValueError("Invalid timestamps")
        if not np.all(np.diff(timestamps) == np.timedelta64(1, "s")):
            raise ValueError("Expected continuous 1-second timestamps")

        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)

        target_index = index + self.sequence_length

        # История не включает целевое наблюдение.
        history = self.features[index:target_index].copy()
        target = self.features[target_index].copy()

        return (
            torch.from_numpy(history),
            torch.from_numpy(target),
        )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]

    for split in ("train", "validation"):
        dataset = ForecastWindowDataset(
            root / f"data/processed/hai/{split}_scaled.npz"
        )
        history, target = dataset[0]

        print(f"\n{split}:")
        print("Windows:", len(dataset))
        print("History shape:", tuple(history.shape))
        print("Target shape:", tuple(target.shape))
        print("First target timestamp:",
              dataset.timestamps[dataset.sequence_length])