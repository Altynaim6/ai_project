"""Document recovered v2 evaluation without changing frozen model or threshold."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from src.config import PROJECT_ROOT as ROOT


def main():
    results = ROOT / "results/hai/transformer_mixed/mixed_v2_full_seed42_run1"
    output = results / "test_evaluation"
    metrics = json.loads((output / "metrics.json").read_text())
    recovery = json.loads((results / "recovery_summary.json").read_text())
    history = pd.read_csv(results / "history.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history.epoch, history.train_loss, "o-", label="Train (dropout enabled)")
    ax.plot(history.epoch, history.validation_loss, "o-", label="Full validation")
    ax.axvline(recovery["best_epoch"], color="grey", ls="--", label="Best saved checkpoint")
    ax.set(xlabel="Completed epoch", ylabel="Mean MSE + CE", title="HAI v2: interrupted after 22 completed epochs")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "figures/training_curves.png", dpi=150)
    plt.close(fig)
    text = [
        "# HAI v2: оценка лучшей сохранённой модели на test2", "",
        "## Состояние обучения",
        "Обучение прервалось после 22 завершённых эпох. Штатный early stopping не завершился; после лучшей эпохи было четыре завершённых эпохи без улучшения. Обучение заново не запускалось.",
        f"Лучшая сохранённая эпоха — {recovery['best_epoch']}; validation loss — {recovery['best_validation_loss']:.8f}.",
        f"После восстановления повторно проверена полная validation. Порог {metrics['threshold']:.8f} рассчитан как 99-й процентиль на отдельной нормальной calibration, затем сохранён и зафиксирован до test.",
        f"Calibration FPR: {recovery['calibration_fpr']*100:.4f}%; validation FPR: {recovery['validation_fpr']*100:.4f}%.",
        "", "## Отсутствие пересечения",
        "До обучения проверены хэши архива и распакованного test2, временные метки и точные 59-канальные дубликаты с полными train1/train2/test1. Пересечений не найдено. Test2 начинается 2019-11-04 15:00:00, calibration заканчивается 14:59:59. Первые 30 test-записей используются только как история; окна не захватывают calibration.",
        "Аудит: manifests/hai_mixed_v2_holdout_audit.json. Отсутствие одинаковых записей не доказывает независимость всех физических сценариев.",
        "", "## Метрики test2", "| Метрика | Результат |", "|---|---|",
    ]
    for key in ("accuracy", "precision", "recall", "f1", "fpr", "average_precision"):
        text.append(f"| {key} | {metrics[key]*100:.2f}% |")
    text += ["", f"Оценены {metrics['evaluated_windows']} целевых наблюдений: {metrics['normal_windows']} нормальных и {metrics['attack_windows']} атакующих.",
             f"TP={metrics['tp']}; FP={metrics['fp']}; TN={metrics['tn']}; FN={metrics['fn']}.",
             "", "## Интерпретация",
             "Модель перестала выдавать тревогу на каждой записи: 145076 нормальных наблюдений распознаны правильно, 1905 ошибочно помечены аномальными. FPR=1.30% — доля ложных тревог среди нормальных наблюдений.",
             "Обнаружены 5081 из 5989 атакующих наблюдений (Recall=84.84%), пропущены 908 (15.16%). Это наблюдения внутри размеченных атак, а не число обнаруженных или пропущенных сценариев.",
             "Precision=72.73%: около 73 из 100 предсказанных аномалий соответствуют атакующим наблюдениям. Число FP не равно числу отдельных пользовательских уведомлений: соседние срабатывания могут образовывать один эпизод.",
             "F1=78.32% отражает компромисс precision/recall при сохранённом пороге. Average Precision=87.86% оценивает ранжирование score по PR-кривой и не является Accuracy или F1.",
             "Целевой ориентир FPR <=1% на test не достигнут: фактически 1.30%. Отдельная calibration около 1% не гарантирует тот же процент на новом файле.",
             "", "## Сравнение и ограничения",
             "V1 давал FPR=100% на test1, v2 даёт 1.30% на test2. Это разные тестовые выборки, поэтому такое сравнение не доказывает, что весь эффект вызван scaler: одновременно расширены нормальные режимы обучения, изменено масштабирование и отделена calibration.",
             "Результат — оценка одного seed и лучшего checkpoint прерванного запуска, а не полностью завершённого 50-эпохового эксперимента и не среднее трёх seeds. Нет основания заявлять гарантированное обнаружение zero-day или готовность для промышленного управления.",
             "После просмотра test2 порог, scaler и веса не менялись. Для дальнейшего подбора test2 уже является просмотренным, требуется новый заранее зафиксированный holdout или отдельный явно обозначенный протокол. Результаты относятся к HAI, не SWaT.",
             "", "## Артефакты",
             "metrics.json/csv; test_predictions.csv; figures/confusion_matrix.png; figures/scores_and_attacks.png; figures/precision_recall.png; figures/metrics.png; figures/training_curves.png.",
             "Исходное прерванное обучение не помечено как штатно завершённое. Состояние восстановления: ../recovery_summary.json; checkpoint, preprocessing и порог связаны контрольными суммами."]
    (output / "experiment_report.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    metrics["training_status"] = "interrupted_after_22_completed_epochs"
    metrics["evaluation_scope"] = "best_saved_checkpoint; early stopping not completed"
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(output / "metrics.csv", index=False)
    print(output / "experiment_report.md")


if __name__ == "__main__":
    main()
