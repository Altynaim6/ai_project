"""Evaluate a frozen mixed Transformer and threshold; never fit on test data."""
import argparse
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, precision_recall_curve)
from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT
from src.mixed_forecast_loss import mixed_forecast_scores
from src.mixed_window_dataset import MixedForecastWindowDataset
from src.models.transformer_mixed import MixedForecastTransformer
from src.models.forecast_factory import create_forecast_model
from src.preprocess_hai_mixed import transform_frame
from src.train_transformer_mixed import file_hash, write_json


def load_frozen_run(model_dir):
    model_dir = Path(model_dir)
    calibration = json.loads((model_dir / "threshold.json").read_text(encoding="utf-8"))
    config = json.loads((model_dir / "run_config.json").read_text(encoding="utf-8"))
    if config.get("run_type") != "full_training" or calibration.get("run_type") != "full_training":
        raise ValueError("This is a smoke-test run. Complete full training before final test evaluation.")
    if calibration.get("decision_rule") != "score > threshold":
        raise ValueError("Unsupported saved decision rule")
    threshold = calibration["threshold"]
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Invalid saved threshold")
    for name, hash_key in (("best_model.pt", "model_sha256"),
                           ("preprocessing.joblib", "preprocessing_sha256")):
        if file_hash(model_dir / name) != calibration[hash_key]:
            raise ValueError(f"Hash mismatch: {name}")
    saved = torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=True)
    bundle = joblib.load(model_dir / "preprocessing.joblib")
    if (saved.get("run_type") != "full_training"
            or saved["epoch"] != calibration["best_epoch"]
            or saved["preprocessing_sha256"] != calibration["preprocessing_sha256"]
            or saved["schema_sha256"] != bundle["schema_sha256"]
            or saved["numeric_columns"] != bundle["numeric_columns"]
            or saved["binary_columns"] != bundle["binary_columns"]):
        raise ValueError("Checkpoint, threshold and preprocessing do not belong to one run")
    kind = saved.get("model_kind", "transformer_mixed")
    if kind != config.get("model_kind", "transformer_mixed") or kind != calibration.get("model_kind", "transformer_mixed"):
        raise ValueError("Checkpoint architecture differs from run/threshold")
    if kind == "hybrid_mixed":
        if saved["model_config"] != config["model_config"]:
            raise ValueError("Hybrid graph/model configuration differs")
        inputs = bundle["numeric_columns"]+bundle["binary_columns"]
        mc = saved["model_config"]
        if [inputs[i] for i in mc["graph_to_input"]] != mc["node_order"]:
            raise ValueError("Graph channel order differs from preprocessing")
    model = create_forecast_model(kind, saved["model_config"])
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    return model, bundle, calibration, saved


def validate_test(frame, bundle, preprocessing_manifest, sequence_length):
    expected_features = bundle["numeric_columns"] + bundle["binary_columns"]
    labels = ["attack", "attack_P1", "attack_P2", "attack_P3"]
    expected = {"time", *expected_features, *labels}
    if (len(frame) <= sequence_length or not frame.columns.is_unique
            or len(frame.columns) != len(expected) or set(frame.columns) != expected):
        raise ValueError("Unexpected test schema or insufficient observations")
    timestamps = pd.to_datetime(frame["time"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    if timestamps.isna().any() or not timestamps.diff().iloc[1:].eq(pd.Timedelta(seconds=1)).all():
        raise ValueError("Expected one continuous, ordered 1-second test segment")
    if not frame[labels].isin([0, 1]).all().all():
        raise ValueError("Invalid test attack labels")
    end = max(pd.Timestamp(info["end"]) for info in preprocessing_manifest["splits"].values())
    if timestamps.iloc[0] <= end:
        raise ValueError("Test overlaps or precedes the validation split")
    return timestamps


def calculate_metrics(labels, scores, threshold):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if (labels.ndim != 1 or labels.shape != scores.shape or not len(labels)
            or not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all()):
        raise ValueError("Invalid labels or scores")
    predictions = (scores > threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else None
    fpr = float(fp / (fp + tn)) if fp + tn else None
    f1 = float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0
    both_classes = len(np.unique(labels)) == 2
    return {
        "evaluated_windows": len(labels), "normal_windows": int(tn + fp),
        "attack_windows": int(tp + fn), "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp), "accuracy": float(accuracy_score(labels, predictions)),
        "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
        "average_precision": float(average_precision_score(labels, scores)) if both_classes else None,
        "threshold": float(threshold), "decision_rule": "score > threshold",
        "metric_unit": "target observation, no point adjustment",
        "undefined_policy": "Precision and F1 use zero_division=0; recall/FPR/AP are null when required classes are absent.",
        "pr_auc_definition": "Average precision (step-weighted), not trapezoidal PR area",
    }, predictions


def plot_results(frame, metrics, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(6, 5))
    display = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                    color="white" if matrix[row, col] > matrix.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "Attack"],
           yticklabels=["Normal", "Attack"], xlabel="Predicted", ylabel="True",
           title="Test confusion matrix: frozen threshold")
    fig.colorbar(display, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    timestamps = pd.to_datetime(frame["timestamp"])
    axes[0].plot(timestamps, frame["score"], lw=.5, label="Score")
    axes[0].axhline(metrics["threshold"], color="red", ls="--", label="Saved threshold")
    axes[0].set_yscale("symlog", linthresh=.01)
    axes[0].set_ylim(bottom=0)
    axes[0].set(ylabel="MSE + CE (symlog)", title="Test scores and ground truth")
    axes[0].legend()
    axes[1].step(timestamps, frame["label"], where="post", label="True attack", lw=1)
    axes[1].step(timestamps, frame["prediction"], where="post", label="Predicted anomaly", lw=.7, alpha=.7)
    axes[1].set(ylabel="State", xlabel="Timestamp", yticks=[0, 1])
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output_dir / "scores_and_attacks.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    if metrics["average_precision"] is not None:
        precision, recall, _ = precision_recall_curve(frame["label"], frame["score"])
        ax.step(recall, precision, where="post", label=f"AP={metrics['average_precision']:.4f}")
        ax.axhline(frame["label"].mean(), color="grey", ls="--", label="Attack prevalence")
        ax.scatter([metrics["recall"]], [metrics["precision"]], color="red", label="Saved threshold", zorder=3)
        ax.legend()
    else:
        ax.text(.5, .5, "PR curve unavailable: one true class", ha="center")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.05), title="Test precision-recall curve")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall.png", dpi=150)
    plt.close(fig)

    names = ["accuracy", "precision", "recall", "f1", "fpr"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, [metrics[name] if metrics[name] is not None else 0 for name in names])
    for bar, name in zip(bars, names):
        value = metrics[name]
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + .015,
                "N/A" if value is None else f"{value:.4f}", ha="center")
    ax.set(ylim=(0, 1.12), ylabel="Fraction", title="Test metrics at saved threshold (lower FPR is better)")
    fig.tight_layout()
    fig.savefig(output_dir / "metrics.png", dpi=150)
    plt.close(fig)


def evaluate(model_dir, test_path, output_dir, batch_size=256, device="cpu"):
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    model_dir, test_path, output_dir = map(Path, (model_dir, test_path, output_dir))
    # Validate the frozen run before opening test data.
    model, bundle, calibration, checkpoint = load_frozen_run(model_dir)
    if output_dir.exists():
        raise FileExistsError("Evaluation directory exists; select a new output directory")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    manifest = json.loads((model_dir / "preprocessing_manifest.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(test_path, sep=";")
    length = checkpoint["model_config"]["sequence_length"]
    validate_test(frame, bundle, manifest, length)
    numeric, binary = transform_frame(frame, bundle)
    output_dir.mkdir(parents=True, exist_ok=False)
    model.to(device)
    score_parts, numeric_parts = [], []
    with tempfile.TemporaryDirectory(prefix=".test-input-", dir=output_dir) as temporary:
        prepared = Path(temporary) / "test.npz"
        np.savez_compressed(
            prepared, numeric=numeric, binary=binary,
            timestamps=frame["time"].to_numpy(dtype=str), labels=frame["attack"].to_numpy(dtype=np.int64),
            numeric_columns=np.array(bundle["numeric_columns"]), binary_columns=np.array(bundle["binary_columns"]),
            schema_sha256=np.array(bundle["schema_sha256"]),
        )
        dataset = MixedForecastWindowDataset(prepared, length)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
        started = time.perf_counter()
        with torch.inference_mode():
            for batch_index, (history, numeric_target, binary_target) in enumerate(loader):
                history, numeric_target, binary_target = (v.to(device) for v in (history, numeric_target, binary_target))
                prediction, logits = model(history)
                scores = mixed_forecast_scores(prediction, logits, numeric_target, binary_target)
                score_parts.append(scores.cpu().numpy())
                numeric_parts.append((prediction - numeric_target).square().mean(dim=1).cpu().numpy())
                if (batch_index + 1) % 100 == 0:
                    print(f"Test: {min((batch_index+1)*batch_size, len(dataset))}/{len(dataset)} windows", flush=True)
        seconds = time.perf_counter() - started
    scores = np.concatenate(score_parts)
    numeric_scores = np.concatenate(numeric_parts)
    labels = frame["attack"].to_numpy(dtype=np.int64)[length:]
    if len(scores) != len(frame) - length:
        raise RuntimeError("Not all test windows were evaluated")
    metrics, predictions = calculate_metrics(labels, scores, calibration["threshold"])
    results = pd.DataFrame({
        "timestamp": frame["time"].iloc[length:].to_numpy(), "label": labels,
        "score": scores, "numeric_mse": numeric_scores,
        "binary_ce": scores - numeric_scores, "prediction": predictions,
        "threshold": calibration["threshold"],
    })
    results.to_csv(output_dir / "test_predictions.csv", index=False)
    metrics.update(
        source_rows=len(frame), warmup_rows_excluded=length,
        best_epoch=checkpoint["epoch"], run_type=checkpoint["run_type"],
        model_sha256=calibration["model_sha256"],
        preprocessing_sha256=calibration["preprocessing_sha256"],
        threshold_file_sha256=file_hash(model_dir / "threshold.json"),
        test_sha256=file_hash(test_path), evaluated_utc=datetime.now(timezone.utc).isoformat(),
        batch_size=batch_size, device=str(device), evaluation_loop_seconds=seconds,
        tuning_on_test=False,
        model_kind=checkpoint.get("model_kind", "transformer_mixed"),
    )
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
    write_json(output_dir / "metrics.json", metrics)
    plot_results(results, metrics, output_dir / "figures")
    (output_dir / "README.md").write_text(
        "# Frozen test evaluation\n\nAll decisions use the saved validation threshold. "
        "The first L observations are warmup and excluded. "
        "Metrics are point-wise; no point adjustment. Average precision uses step-weighted AP. "
        "Evaluation loop duration is not a batch=1 inference latency benchmark. "
        "Do not use these test results to retune the same experiment.\n", encoding="utf-8"
    )
    write_json(output_dir / "evaluation_status.json", {"status": "completed", "evaluated_windows": len(scores)})
    print(json.dumps(metrics, indent=2), flush=True)
    print("Saved:", output_dir)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True, help="Completed full-training run")
    parser.add_argument("--test-file", type=Path, default=PROJECT_ROOT / "data/raw/hai/test1.csv")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if not args.run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.run_name):
        parser.error("Invalid run name")
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    evaluate(
        PROJECT_ROOT / f"artifacts/hai/transformer_mixed/{args.run_name}", args.test_file,
        PROJECT_ROOT / f"results/hai/transformer_mixed/{args.run_name}/test_evaluation",
        args.batch_size, args.device,
    )


if __name__ == "__main__":
    main()
