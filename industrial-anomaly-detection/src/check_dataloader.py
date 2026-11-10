import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT, load_config
from src.window_dataset import ForecastWindowDataset


def main():
    config = load_config()

    seed = config["experiment"]["seeds"][0]
    sequence_length = config["experiment"]["sequence_length"]
    batch_size = config["training"]["batch_size"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_directory = PROJECT_ROOT / "data" / "processed" / "hai"

    train_dataset = ForecastWindowDataset(
        data_directory / "train_scaled.npz",
        sequence_length=sequence_length,
    )
    validation_dataset = ForecastWindowDataset(
        data_directory / "validation_scaled.npz",
        sequence_length=sequence_length,
    )

    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    for name, loader in (
        ("Train", train_loader),
        ("Validation", validation_loader),
    ):
        history, target = next(iter(loader))

        assert history.shape[1:] == (
            sequence_length,
            len(train_dataset.feature_columns),
        )
        assert target.shape == (
            history.shape[0],
            len(train_dataset.feature_columns),
        )
        assert torch.isfinite(history).all()
        assert torch.isfinite(target).all()

        print(f"\n{name}:")
        print("Batches:", len(loader))
        print("History batch:", tuple(history.shape))
        print("Target batch:", tuple(target.shape))
        print("Data type:", history.dtype)

    print("\nDataLoader check completed")


if __name__ == "__main__":
    main()