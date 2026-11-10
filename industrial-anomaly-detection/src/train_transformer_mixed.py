"""Train mixed forecasts, select by full validation, calibrate best checkpoint."""
import argparse
import hashlib
import json
import platform
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT, load_config
from src.mixed_forecast_loss import mixed_forecast_scores
from src.mixed_window_dataset import MixedForecastWindowDataset
from src.models.transformer_mixed import MixedForecastTransformer
from src.models.forecast_factory import create_forecast_model


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def run_epoch(model, loader, device, optimizer=None, max_train_batches=None,
              collect_scores=False, progress_every=100):
    """Validation is always complete; mean losses are weighted by sample count."""
    training = optimizer is not None
    if max_train_batches is not None and (not training or max_train_batches < 1):
        raise ValueError("A positive batch limit is allowed only for training")
    model.train(training)
    sums = np.zeros(3, dtype=np.float64)
    count, batches = 0, 0
    collected = []
    started = time.perf_counter()
    with torch.set_grad_enabled(training):
        for batch_index, (history, numeric, binary) in enumerate(loader):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            history, numeric, binary = (value.to(device) for value in (history, numeric, binary))
            if training:
                optimizer.zero_grad(set_to_none=True)
            predicted, logits = model(history)
            scores = mixed_forecast_scores(predicted, logits, numeric, binary)
            numeric_scores = (predicted - numeric).square().mean(dim=1)
            loss = scores.mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            sums += np.array([
                scores.detach().double().sum().item(),
                numeric_scores.detach().double().sum().item(),
                (scores - numeric_scores).detach().double().sum().item(),
            ])
            count += len(history)
            batches += 1
            if collect_scores:
                collected.append(scores.detach().cpu().numpy())
            if progress_every and batches % progress_every == 0:
                print(f"  {'train' if training else 'validation'}: {batches} batches, "
                      f"{count} windows, loss={sums[0]/count:.6f}", flush=True)
    if not count:
        raise ValueError("No windows processed")
    if not training and count != len(loader.dataset):
        raise ValueError("Validation must include every window exactly once")
    metrics = dict(zip(("loss", "numeric_mse", "binary_ce"), (sums / count).tolist()))
    metrics.update(windows=count, batches=batches, seconds=time.perf_counter() - started)
    return metrics, np.concatenate(collected) if collect_scores else None


def calibrate_scores(scores, quantile=0.99):
    scores = np.asarray(scores)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Expected finite, non-empty score vector")
    if not 0 < quantile < 1:
        raise ValueError("Quantile must be between 0 and 1")
    threshold = float(np.quantile(scores, quantile, method="linear"))
    return {
        "method": "normal_validation_quantile", "quantile": quantile,
        "quantile_method": "linear", "threshold": threshold,
        "decision_rule": "score > threshold",
        "validation_windows": len(scores),
        "validation_fpr": float(np.mean(scores > threshold)),
        "score": "mean_numeric_MSE + mean_binary_CE",
        "note": "In-sample calibration FPR; not an independent test estimate. "
                "The same validation split selects the checkpoint and the threshold.",
    }


def run_training(data_dir, preprocessing_dir, model_dir, results_dir,
                 training_config, model_config, seed=42, device="cpu",
                 max_train_batches=None, progress_every=100, model_kind="transformer_mixed"):
    data_dir, preprocessing_dir = Path(data_dir), Path(preprocessing_dir)
    model_dir, results_dir = Path(model_dir), Path(results_dir)
    epochs = training_config["max_epochs"]
    patience = training_config["early_stopping_patience"]
    batch_size = training_config["batch_size"]
    eval_batch_size = training_config.get("validation_batch_size", 256)
    if min(epochs, patience, batch_size, eval_batch_size) < 1 or training_config["learning_rate"] <= 0:
        raise ValueError("Training settings must be positive")
    if max_train_batches is not None and max_train_batches < 1:
        raise ValueError("max_train_batches must be positive")
    if model_dir.exists() or results_dir.exists():
        raise FileExistsError("Use new run directories; existing runs are never overwritten")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        torch.cuda.manual_seed_all(seed)

    split_names = ["train", "validation"]
    if (data_dir / "calibration.npz").exists():
        split_names.append("calibration")
    datasets = {split: MixedForecastWindowDataset(
        data_dir / f"{split}.npz", model_config["sequence_length"]
    ) for split in split_names}
    train, validation = datasets["train"], datasets["validation"]
    bundle = joblib.load(preprocessing_dir / "preprocessing.joblib")
    bundle_hash = file_hash(preprocessing_dir / "preprocessing.joblib")
    for dataset in datasets.values():
        if (dataset.numeric_columns != bundle["numeric_columns"]
                or dataset.binary_columns != bundle["binary_columns"]
                or dataset.schema_sha256 != bundle["schema_sha256"]):
            raise ValueError("Dataset/preprocessing schemas differ")
        if not np.all(dataset.labels == 0):
            raise ValueError("Forecast training/calibration requires normal-only splits")
        if dataset.preprocessing_sha256 is not None and dataset.preprocessing_sha256 != bundle_hash:
            raise ValueError("Prepared arrays belong to another preprocessing bundle")
    if (model_config["num_numeric"] != len(train.numeric_columns)
            or model_config["num_binary"] != len(train.binary_columns)):
        raise ValueError("Model dimensions differ from data")
    if train.timestamps[-1] >= validation.timestamps[0]:
        raise ValueError("Train/validation timestamps overlap")
    if "calibration" in datasets and validation.timestamps[-1] >= datasets["calibration"].timestamps[0]:
        raise ValueError("Validation/calibration timestamps overlap")

    loaders = {
        "train": DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(seed)),
        "validation": DataLoader(validation, batch_size=eval_batch_size,
                                 shuffle=False, num_workers=0, drop_last=False),
    }
    if "calibration" in datasets:
        loaders["calibration"] = DataLoader(datasets["calibration"], batch_size=eval_batch_size,
                                            shuffle=False, num_workers=0, drop_last=False)
    model = create_forecast_model(model_kind, model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config["learning_rate"])
    model_dir.mkdir(parents=True, exist_ok=False)
    results_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(preprocessing_dir / "preprocessing.joblib", model_dir / "preprocessing.joblib")
    shutil.copy2(preprocessing_dir / "manifest.json", model_dir / "preprocessing_manifest.json")
    run_type = "smoke_test" if max_train_batches is not None else "full_training"
    run_config = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "run_type": run_type, "seed": seed, "device": str(device), "model_kind": model_kind,
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "python_version": platform.python_version(), "platform": platform.platform(),
        "torch_threads": torch.get_num_threads(),
        "model_config": model_config, "training_config": training_config,
        "max_train_batches": max_train_batches, "validation": "all windows, ordered",
        "numeric_columns": train.numeric_columns, "binary_columns": train.binary_columns,
        "schema_sha256": train.schema_sha256,
        "data_sha256": {split: file_hash(data_dir / f"{split}.npz") for split in datasets},
        "preprocessing_sha256": file_hash(model_dir / "preprocessing.joblib"),
        "loss_group_weights": {"numeric": 1.0, "binary": 1.0},
    }
    write_json(model_dir / "run_config.json", run_config)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs} ({run_type})", flush=True)
        train_metrics, _ = run_epoch(model, loaders["train"], device, optimizer,
                                     max_train_batches, progress_every=progress_every)
        validation_metrics, _ = run_epoch(model, loaders["validation"], device,
                                          progress_every=progress_every)
        improved = validation_metrics["loss"] < best_loss
        if improved:
            best_loss, best_epoch, stale = validation_metrics["loss"], epoch, 0
            checkpoint = {
                "model_state_dict": model.state_dict(), "model_config": model.model_config,
                "model_kind": model_kind,
                "numeric_columns": train.numeric_columns, "binary_columns": train.binary_columns,
                "schema_sha256": train.schema_sha256,
                "preprocessing_sha256": run_config["preprocessing_sha256"],
                "seed": seed, "epoch": epoch, "validation_loss": best_loss,
                "run_type": run_type,
            }
            temporary = model_dir / "best_model.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(model_dir / "best_model.pt")
        else:
            stale += 1
        row = {"epoch": epoch, "best": improved}
        for prefix, metrics in (("train", train_metrics), ("validation", validation_metrics)):
            row.update({f"{prefix}_{key}": value for key, value in metrics.items()})
        history.append(row)
        pd.DataFrame(history).to_csv(results_dir / "history.csv", index=False)
        write_json(results_dir / "history.json", history)
        print(f"  train={train_metrics['loss']:.6f}; full validation={validation_metrics['loss']:.6f}; "
              f"best epoch={best_epoch}; patience={stale}/{patience}", flush=True)
        if stale >= patience:
            print("Early stopping", flush=True)
            break

    # Reload exactly the selected checkpoint, not the last epoch.
    saved = torch.load(model_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state_dict"])
    metrics, scores = run_epoch(model, loaders["validation"], device,
                                collect_scores=True, progress_every=progress_every)
    if not np.isclose(metrics["loss"], best_loss, rtol=1e-5, atol=1e-6):
        raise RuntimeError("Restored checkpoint validation differs from selected checkpoint")
    calibration_split = "calibration" if "calibration" in datasets else "validation"
    calibration_scores = scores
    if calibration_split == "calibration":
        _, calibration_scores = run_epoch(model, loaders["calibration"], device,
                                          collect_scores=True, progress_every=progress_every)
    calibration = calibrate_scores(calibration_scores, training_config.get("threshold_quantile", .99))
    calibration["calibration_split"] = calibration_split
    calibration["calibration_windows"] = len(calibration_scores)
    calibration["validation_windows"] = len(scores)
    calibration["calibration_fpr"] = calibration["validation_fpr"]
    calibration["validation_fpr"] = float(np.mean(scores > calibration["threshold"]))
    if calibration_split == "calibration":
        calibration["method"] = "normal_calibration_quantile"
        calibration["note"] = "Checkpoint selected on validation; threshold fitted on a separate normal calibration split. Test never used."
    calibration.update(
        model_sha256=file_hash(model_dir / "best_model.pt"),
        preprocessing_sha256=run_config["preprocessing_sha256"],
        validation_data_sha256=run_config["data_sha256"]["validation"],
        calibration_data_sha256=run_config["data_sha256"][calibration_split],
        best_epoch=best_epoch, run_type=run_type, model_kind=model_kind,
    )
    write_json(model_dir / "threshold.json", calibration)
    pd.DataFrame({
        "timestamp": validation.timestamps[validation.target_indices],
        "label": validation.labels[validation.target_indices],
        "score": scores,
        "prediction": (scores > calibration["threshold"]).astype(np.int64),
    }).to_csv(results_dir / "validation_scores.csv", index=False)
    if calibration_split == "calibration":
        calibration_dataset = datasets["calibration"]
        pd.DataFrame({
            "timestamp": calibration_dataset.timestamps[calibration_dataset.target_indices],
            "label": calibration_dataset.labels[calibration_dataset.target_indices],
            "score": calibration_scores,
            "prediction": (calibration_scores > calibration["threshold"]).astype(np.int64),
        }).to_csv(results_dir / "calibration_scores.csv", index=False)
    summary = {
        "status": "completed", "run_type": run_type, "epochs_completed": len(history),
        "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "validation_windows": len(scores), "threshold": calibration["threshold"],
        "validation_fpr": calibration["validation_fpr"],
        "calibration_split": calibration_split,
        "calibration_windows": len(calibration_scores),
        "calibration_fpr": calibration["calibration_fpr"],
        "stopped_early": stale >= patience,
        "test_used": False,
    }
    write_json(results_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    print("Model:", model_dir, "\nResults:", results_dir, flush=True)
    return summary


def main():
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=config["training"]["max_epochs"])
    parser.add_argument("--patience", type=int, default=config["training"]["early_stopping_patience"])
    parser.add_argument("--batch-size", type=int, default=config["training"]["batch_size"])
    parser.add_argument("--validation-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=config["experiment"]["seeds"][0])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, help="Smoke test only; validation remains complete")
    parser.add_argument("--run-name", help="New run name, single directory component")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/processed/hai_mixed")
    parser.add_argument("--preprocessing-dir", type=Path, default=PROJECT_ROOT / "artifacts/hai/preprocessing_mixed")
    args = parser.parse_args()
    name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if not name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in name):
        parser.error("run-name must contain only letters, digits, underscore or hyphen")
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    training = dict(config["training"], max_epochs=args.epochs,
                    early_stopping_patience=args.patience, batch_size=args.batch_size,
                    validation_batch_size=args.validation_batch_size)
    run_training(
        args.data_dir,
        args.preprocessing_dir,
        PROJECT_ROOT / f"artifacts/hai/transformer_mixed/{name}",
        PROJECT_ROOT / f"results/hai/transformer_mixed/{name}",
        training, dict(num_numeric=55, num_binary=4,
                       sequence_length=config["experiment"]["sequence_length"],
                       d_model=64, nhead=4, num_layers=2, dropout=0.1),
        seed=args.seed, device=args.device, max_train_batches=args.max_train_batches,
    )


if __name__ == "__main__":
    main()
