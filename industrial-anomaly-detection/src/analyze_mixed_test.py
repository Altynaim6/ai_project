"""Post-test diagnostics only; does not change model, threshold or preprocessing."""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.config import PROJECT_ROOT
from src.evaluate_transformer_mixed import load_frozen_run
from src.preprocess_hai_mixed import transform_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.run_name):
        parser.error("Invalid run name")
    torch.set_num_threads(4)
    model_dir = PROJECT_ROOT / "artifacts/hai/transformer_mixed" / args.run_name
    results = PROJECT_ROOT / "results/hai/transformer_mixed" / args.run_name
    output = results / "test_evaluation"
    metrics = json.loads((output / "metrics.json").read_text())
    summary = json.loads((results / "summary.json").read_text())
    predictions = pd.read_csv(output / "test_predictions.csv")
    model, bundle, calibration, checkpoint = load_frozen_run(model_dir)
    frame = pd.read_csv(PROJECT_ROOT / "data/raw/hai/test1.csv", sep=";")
    numeric, binary = transform_frame(frame, bundle)
    inputs = np.concatenate([numeric, binary], axis=1).astype(np.float32)
    length = checkpoint["model_config"]["sequence_length"]
    sums = np.zeros((2, len(bundle["numeric_columns"])), dtype=np.float64)
    counts = np.zeros(2, dtype=np.int64)
    labels = frame["attack"].to_numpy()
    with torch.inference_mode():
        for start in range(length, len(frame), 256):
            targets = np.arange(start, min(start + 256, len(frame)))
            histories = np.stack([inputs[t-length:t] for t in targets])
            forecast, _ = model(torch.from_numpy(histories))
            error = (forecast.numpy().astype(np.float64) - numeric[targets]) ** 2
            for label in (0, 1):
                selected = labels[targets] == label
                sums[label] += error[selected].sum(axis=0)
                counts[label] += selected.sum()
    channels = pd.DataFrame({
        "channel": bundle["numeric_columns"],
        "normal_test_mse": sums[0] / counts[0],
        "attack_test_mse": sums[1] / counts[1],
        "train_mean": bundle["scaler"].mean_,
        "train_std": np.sqrt(bundle["scaler"].var_),
        "constant_on_train": bundle["scaler"].var_ == 0,
        "normal_test_raw_median": frame.loc[frame["attack"].eq(0), bundle["numeric_columns"]].median().to_numpy(),
    }).sort_values("normal_test_mse", ascending=False)
    channels["normal_score_contribution"] = channels["normal_test_mse"] / len(bundle["numeric_columns"])
    channels.to_csv(output / "channel_diagnostics.csv", index=False)
    history = pd.read_csv(results / "history.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history["epoch"], history["train_loss"], "o-", label="Train (dropout enabled)")
    ax.plot(history["epoch"], history["validation_loss"], "o-", label="Full validation")
    ax.axvline(summary["best_epoch"], color="grey", ls="--", label="Best checkpoint")
    ax.set(xlabel="Epoch", ylabel="Mean MSE + CE", title="Full mixed Transformer training")
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "figures/training_curves.png", dpi=150)
    plt.close(fig)
    train_seconds = float(history["train_seconds"].sum() + history["validation_seconds"].sum())
    normal_scores = predictions.loc[predictions["label"].eq(0), "score"]
    lines = [
        "# Результат полного обучения и test-оценки HAI 20.07", "",
        f"Запуск: `{args.run_name}`. Обучено {summary['epochs_completed']} эпох; лучшая — {summary['best_epoch']}. Early stopping после 5 эпох без улучшения.",
        f"Полная validation loss: {summary['best_validation_loss']:.6f}. Порог: {calibration['threshold']:.8f}, правило score > threshold.",
        f"Суммарное время train + validation по журналу: {train_seconds/60:.1f} мин (без подготовки и экспорта).", "",
        "| Метрика | Результат |", "|---|---|",
    ]
    for key in ("accuracy", "precision", "recall", "f1", "fpr", "average_precision"):
        value = metrics[key]
        lines.append(f"| {key} | {value:.6f} |" if value is not None else f"| {key} | N/A |")
    lines += ["", f"TP={metrics['tp']}, FP={metrics['fp']}, TN={metrics['tn']}, FN={metrics['fn']}. Оценены {metrics['evaluated_windows']} целей, первые {length} исходных строк исключены как warmup.",
              "", "## Интерпретация",
              "Все test-цели получили метку Anomaly. Recall=100% здесь не означает полезную детекцию: FPR=100%, нормальная работа не распознаётся.",
              f"Минимальный score нормальных test-целей {normal_scores.min():.4f}, медиана {normal_scores.median():.4f}, тогда как порог {calibration['threshold']:.4f}.",
              "Порог подобран на нормальной validation из того же train1; достигнутые там около 1% ложных тревог не переносятся на другую запись test1.",
              "Наблюдается сдвиг значений/режимов между train1 и test1. Это подтверждается таблицей ниже; точная физическая причина сдвига без дополнительных журналов не установлена.",
              "", "## Наибольшие числовые ошибки на нормальном test",
              "| Канал | MSE канала | Вклад в средний numeric score | Train mean | Normal test median |",
              "|---|---|---|---|---|"]
    for _, row in channels.head(8).iterrows():
        lines.append(f"| {row['channel']} | {row['normal_test_mse']:.4f} | {row['normal_score_contribution']:.4f} | {row['train_mean']:.4f} | {row['normal_test_raw_median']:.4f} |")
    lines += ["", "Диагностика выполнена после test-оценки и не использовалась для изменения порога, весов модели или scaler. Все артефакты исходного эксперимента сохранены.",
              "", "## Дальнейшая работа",
              "Считать этот запуск отрицательным baseline-результатом. Проверить покрытие нормальных режимов на дополнительных train-данных (например, train2), сравнить с persistence-baseline и train-only вариантами обработки уставок/низкодисперсных каналов. После просмотра test1 он уже не является нетронутым holdout для новой итерации: закрепить отдельный будущий holdout до настройки. Не подбирать новый порог по test1 и не fit scaler на test.",
              "Результаты относятся к временному HAI-протоколу, не к SWaT, и не подтверждают обнаружение zero-day атак."]
    (output / "experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(channels.head(8).to_string(index=False))
    print("Report:", output / "experiment_report.md")


if __name__ == "__main__":
    main()
