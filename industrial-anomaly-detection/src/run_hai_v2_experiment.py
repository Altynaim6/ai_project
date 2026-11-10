"""Complete frozen v2 training, then evaluate the audited test2 exactly once."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.config import PROJECT_ROOT, load_config
from src.evaluate_transformer_mixed import evaluate
from src.train_transformer_mixed import file_hash, run_training


def main():
    root = PROJECT_ROOT
    audit = json.loads((root / "manifests/hai_mixed_v2_holdout_audit.json").read_text())
    test_path = root / audit["holdout"]
    if not audit["passed"] or file_hash(test_path) != audit["extracted_sha256"]:
        raise ValueError("Test2 audit missing/failed or file changed")
    name = "mixed_v2_full_seed42_run1"
    model_dir = root / "artifacts/hai/transformer_mixed" / name
    results_dir = root / "results/hai/transformer_mixed" / name
    config = load_config()
    torch.set_num_threads(4)
    summary = run_training(
        root / "data/processed/hai_mixed_v2",
        root / "artifacts/hai/preprocessing_hai_mixed_v2",
        model_dir, results_dir, config["training"],
        dict(num_numeric=55, num_binary=4, sequence_length=config["experiment"]["sequence_length"],
             d_model=64, nhead=4, num_layers=2, dropout=.1),
        seed=42, device="cpu",
    )
    # Integrity check after training, before using any test scores.
    if file_hash(test_path) != audit["extracted_sha256"]:
        raise ValueError("Holdout changed during training")
    metrics = evaluate(model_dir, test_path, results_dir / "test_evaluation")
    history = pd.read_csv(results_dir / "history.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history.epoch, history.train_loss, "o-", label="Train (dropout enabled)")
    ax.plot(history.epoch, history.validation_loss, "o-", label="Full validation")
    ax.axvline(summary["best_epoch"], color="grey", ls="--", label="Best checkpoint")
    ax.set(xlabel="Epoch", ylabel="Mean MSE + CE", title="HAI v2 full training")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(results_dir / "test_evaluation/figures/training_curves.png", dpi=150)
    plt.close(fig)
    report = [
        "# HAI v2: полное обучение и независимая test2-оценка", "",
        f"Запуск: {name}. Эпох выполнено: {summary['epochs_completed']}; лучшая: {summary['best_epoch']}.",
        f"Лучшая validation loss: {summary['best_validation_loss']:.6f}.",
        f"Порог {summary['threshold']:.8f} выбран на отдельной нормальной calibration, не на test2.",
        "Проверка до обучения: общих временных меток и точных 59-канальных дубликатов с train1/train2/test1 не найдено. Это не доказывает независимость всех физических сценариев.",
        f"Test2: {metrics['source_rows']} строк, оценено {metrics['evaluated_windows']} целей; первые {metrics['warmup_rows_excluded']} строк — warmup.",
        "", "| Метрика | Значение |", "|---|---|",
    ]
    for key in ("accuracy", "precision", "recall", "f1", "fpr", "average_precision"):
        value = metrics[key]
        report.append(f"| {key} | {value:.6f} |" if value is not None else f"| {key} | N/A |")
    report += ["", f"TP={metrics['tp']}; FP={metrics['fp']}; TN={metrics['tn']}; FN={metrics['fn']}.",
               f"FPR на validation: {summary['validation_fpr']:.6f}; на calibration: {summary['calibration_fpr']:.6f}; на test2: {metrics['fpr']:.6f}.",
               "Метрики поточечные, без point adjustment. Average Precision — ранжирующая AP, а не оценка на выбранном пороге.",
               "", "## Границы выводов",
               "V1 оценивался на test1, V2 — на test2. Прямое сравнение их процентов не изолирует влияние preprocessing: изменились данные, покрытие режимов и split. Подтверждение улучшения требует заранее спланированного сравнения замороженных вариантов на одной выборке.",
               "Для v2 одновременно расширено обучение нормальными режимами train2, введён scale floor и отделена calibration. Приписывать весь эффект только одному изменению нельзя.",
               "Порог, scaler и модель после просмотра test2 не изменялись. Test2 после этой оценки уже не является нетронутым holdout для дальнейшего подбора. Результат относится к HAI, не к SWaT и не является доказательством zero-day обобщения."]
    output = results_dir / "test_evaluation/experiment_report.md"
    output.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("V2 experiment completed. Report:", output, flush=True)


if __name__ == "__main__":
    main()
