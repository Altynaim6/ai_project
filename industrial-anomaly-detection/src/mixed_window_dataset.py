"""Lazy windows, grouped numeric/binary inputs, separate forecast targets."""
import numpy as np
import torch
from torch.utils.data import Dataset


class MixedForecastWindowDataset(Dataset):
    def __init__(self, path, sequence_length=30):
        if not isinstance(sequence_length, int) or sequence_length < 1:
            raise ValueError("Sequence length must be a positive integer")
        with np.load(path, allow_pickle=False) as data:
            self.numeric = data["numeric"].copy()
            self.binary = data["binary"].copy()
            self.timestamps = data["timestamps"].copy()
            self.labels = data["labels"].copy()
            self.numeric_columns = data["numeric_columns"].tolist()
            self.binary_columns = data["binary_columns"].tolist()
            self.schema_sha256 = str(data["schema_sha256"].item())
            self.preprocessing_sha256 = (
                str(data["preprocessing_sha256"].item())
                if "preprocessing_sha256" in data.files else None
            )
        rows = len(self.numeric)
        if self.numeric.shape != (rows, len(self.numeric_columns)):
            raise ValueError("Invalid numeric shape")
        if self.binary.shape != (rows, len(self.binary_columns)):
            raise ValueError("Invalid binary shape")
        if len(self.timestamps) != rows or len(self.labels) != rows or rows <= sequence_length:
            raise ValueError("Invalid lengths or insufficient rows")
        if not np.isfinite(self.numeric).all() or not np.isin(self.binary, [0, 1]).all():
            raise ValueError("Invalid feature values")
        if not np.isin(self.labels, [0, 1]).all():
            raise ValueError("Invalid attack labels")
        times = self.timestamps.astype("datetime64[s]")
        differences = np.diff(times)
        if np.isnat(times).any() or np.any(differences <= np.timedelta64(0, "s")):
            raise ValueError("Timestamps must be valid, strictly increasing and unique")
        boundaries = np.r_[0, np.flatnonzero(differences != np.timedelta64(1, "s")) + 1, rows]
        self.segment_count = len(boundaries) - 1
        starts = [np.arange(start, end - sequence_length, dtype=np.int64)
                  for start, end in zip(boundaries[:-1], boundaries[1:])
                  if end - start > sequence_length]
        if not starts:
            raise ValueError("No segment is long enough for a complete window")
        self.window_start_indices = np.concatenate(starts)
        self.target_indices = self.window_start_indices + sequence_length
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.window_start_indices)

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = self.window_start_indices[index]
        target = start + self.sequence_length
        # Fixed input order: 55 scaled numeric channels, then four raw 0/1 states.
        history = np.concatenate(
            [self.numeric[start:target], self.binary[start:target]], axis=1
        ).astype(np.float32)
        return (
            torch.from_numpy(history),
            torch.tensor(self.numeric[target], dtype=torch.float32),
            torch.tensor(self.binary[target], dtype=torch.long),
        )
