import json

import matplotlib.pyplot as plt
import pandas as pd

from src.config import PROJECT_ROOT


def main():
    model_dir = PROJECT_ROOT / "artifacts/hai/transformer_smoke"
    results_dir = PROJECT_ROOT / "results/hai/transformer_smoke"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    history = pd.DataFrame(
        json.loads((model_dir / "history.json").read_text(encoding="utf-8"))
    )
    calibration = json.loads(
        (model_dir / "threshold.json").read_text(encoding="utf-8")
    )
    scores = pd.read_csv(
        results_dir / "validation_scores.csv",
        parse_dates=["timestamp"],
    )
    threshold = calibration["threshold"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 11))

    axes[0].plot(
        history["epoch"], history["train_loss"], "o-", label="Train"
    )
    axes[0].plot(
        history["epoch"],
        history["validation_loss"],
        "o-",
        label="Validation subset",
    )
    axes[0].set(
        title="Smoke training: 100 batches per epoch",
        xlabel="Epoch",
        ylabel="MSE",
    )
    axes[0].set_xticks(history["epoch"])
    axes[0].legend()

    axes[1].plot(
        scores["timestamp"], scores["score"], linewidth=0.6
    )
    axes[1].axhline(
        threshold, color="red", linestyle="--", label="Threshold"
    )
    axes[1].set(
        title="Normal validation: forecast error over time",
        xlabel="Timestamp",
        ylabel="Anomaly score (MSE)",
    )
    axes[1].legend()

    axes[2].hist(scores["score"], bins=100)
    axes[2].axvline(
        threshold, color="red", linestyle="--", label="99th percentile"
    )
    axes[2].set(
        title="Validation score distribution",
        xlabel="Anomaly score (MSE)",
        ylabel="Number of observations",
    )
    axes[2].legend()

    for axis in axes:
        axis.grid(alpha=0.2)

    fig.tight_layout()
    output = figures_dir / "training_and_validation.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)

    print("Saved:", output)


if __name__ == "__main__":
    main()