import copy
import json

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT, load_config
from src.models.transformer import ForecastTransformer
from src.window_dataset import ForecastWindowDataset


def run_epoch(model, loader, criterion, optimizer=None, max_batches=100):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0

    with torch.set_grad_enabled(training):
        for batch_index, (history, target) in enumerate(loader):
            if batch_index >= max_batches:
                break

            if training:
                optimizer.zero_grad(set_to_none=True)

            prediction = model(history)
            loss = criterion(prediction, target)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                    error_if_nonfinite=True,
                )
                optimizer.step()

            total_loss += loss.item() * len(history)
            total_samples += len(history)

    if total_samples == 0:
        raise RuntimeError("No samples processed")

    return total_loss / total_samples


def main():
    config = load_config()
    seed = config["experiment"]["seeds"][0]
    torch.manual_seed(seed)

    sequence_length = config["experiment"]["sequence_length"]
    data_dir = PROJECT_ROOT / "data/processed/hai"
    output_dir = PROJECT_ROOT / "artifacts/hai/transformer_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        split: ForecastWindowDataset(
            data_dir / f"{split}_scaled.npz",
            sequence_length=sequence_length,
        )
        for split in ("train", "validation")
    }

    loaders = {
        split: DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=(split == "train"),
            generator=torch.Generator().manual_seed(seed),
            num_workers=0,
        )
        for split, dataset in datasets.items()
    }

    model_config = {
        "num_features": len(datasets["train"].feature_columns),
        "sequence_length": sequence_length,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.1,
    }

    if not (
        datasets["train"].feature_columns
        == datasets["validation"].feature_columns
    ).all():
        raise ValueError("Feature order differs between splits")

    # Пробный запуск выполняется на CPU.
    model = ForecastTransformer(**model_config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )
    criterion = nn.MSELoss()

    history = []
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, 4):
        train_loss = run_epoch(
            model, loaders["train"], criterion, optimizer,
            max_batches=100,
        )
        validation_loss = run_epoch(
            model, loaders["validation"], criterion,
            max_batches=100,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        })

        print(
            f"Epoch {epoch}/3 | "
            f"train={train_loss:.6f} | "
            f"validation={validation_loss:.6f}",
            flush=True,
        )

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())

            torch.save(
                {
                    "model_state_dict": best_state,
                    "model_config": model_config,
                    "feature_columns":
                        datasets["train"].feature_columns.tolist(),
                    "seed": seed,
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "run_type": "smoke_test",
                },
                output_dir / "best_model.pt",
            )

    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    # Проверяем восстановление лучшей модели.
    model.load_state_dict(best_state)
    model.eval()

    checkpoint = torch.load(
        output_dir / "best_model.pt",
        map_location="cpu",
        weights_only=True,
    )
    restored = ForecastTransformer(**checkpoint["model_config"])
    restored.load_state_dict(checkpoint["model_state_dict"])
    restored.eval()

    sample, _ = next(iter(loaders["validation"]))

    with torch.no_grad():
        expected = model(sample)
        actual = restored(sample)

    torch.testing.assert_close(
        actual, expected, rtol=0, atol=1e-5
    )

    print("Save/load check: OK")
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()