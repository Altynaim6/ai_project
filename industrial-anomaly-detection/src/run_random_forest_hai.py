"""Fit frozen RF baselines on development labels; compare identical test2 targets."""
import json
import shutil
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT as ROOT
from src.evaluate_transformer_mixed import calculate_metrics, plot_results, validate_test
from src.models.baseline import (RandomForestBaseline, feature_names, make_features,
                                 sample_training_indices, valid_targets)
from src.prepare_hai_v2 import audit_source
from src.preprocess_hai_mixed import transform_frame
from src.train_transformer_mixed import calibrate_scores, file_hash, write_json


def transformed(frame, bundle):
    numeric, binary = transform_frame(frame, bundle)
    return np.concatenate([numeric, binary], axis=1).astype(np.float32)


def freeze_protocol(config, schema, metadata, preprocessing_dir, artifacts):
    if artifacts.exists():
        raise FileExistsError("RF experiment exists; use a separately versioned protocol")
    artifacts.mkdir(parents=True)
    protocol = dict(config,
        training_rule="Raw train1 + first 70% raw train2, including training attacks; sources windowed separately",
        validation_rule="Next 15% train2, normal-only; reporting only (no parameter search)",
        calibration_rule="Last 15% train2, normal-only; fixed 99th percentile per-model score",
        preprocessing="Frozen v2 normal-train scaler; no refit for RF",
        preprocessing_sha256=file_hash(preprocessing_dir / "preprocessing.joblib"),
        development_sha256={name: info["sha256"] for name, info in metadata["source_audit"].items()},
        label="attack at target observation; attack* columns excluded from features",
        comparison_caveat="RF has training attack labels, Transformer trained only on normal data. Not architecture-only ablation.",
        prior_test_results_seen=True, tuning_on_test=False,
        source_channels=schema["feature_order"],
    )
    write_json(artifacts / "protocol.json", protocol)
    shutil.copy2(preprocessing_dir / "preprocessing.joblib", artifacts / "preprocessing.joblib")
    shutil.copy2(preprocessing_dir / "manifest.json", artifacts / "preprocessing_manifest.json")
    return protocol


def main():
    config = json.loads((ROOT / "configs/random_forest_hai.json").read_text())
    schema = json.loads((ROOT / "configs/hai_20_07_schema.json").read_text())
    preprocessing_dir = ROOT / "artifacts/hai/preprocessing_hai_mixed_v2"
    metadata = json.loads((preprocessing_dir / "manifest.json").read_text())
    artifacts = ROOT / "artifacts/hai/random_forest" / config["protocol_version"]
    results = ROOT / "results/hai/random_forest" / config["protocol_version"]
    if results.exists():
        raise FileExistsError(results)
    protocol = freeze_protocol(config, schema, metadata, preprocessing_dir, artifacts)
    results.mkdir(parents=True)
    bundle = joblib.load(artifacts / "preprocessing.joblib")
    frames = {}
    for source in ("train1", "train2"):
        path = ROOT / f"data/raw/hai/{source}.csv"
        if file_hash(path) != protocol["development_sha256"][source]:
            raise ValueError("Development source changed")
        frames[source] = audit_source(path, schema)
    n = len(frames["train2"])
    first, second = int(n*.70), int(n*.85)
    train_frames = [frames["train1"], frames["train2"].iloc[:first]]
    validation = frames["train2"].iloc[first:second].reset_index(drop=True)
    calibration = frames["train2"].iloc[second:].reset_index(drop=True)
    length = config["sequence_length"]
    if not validation.attack.eq(0).all() or not calibration.attack.eq(0).all():
        raise ValueError("Fixed normal validation/calibration contains attacks")
    for split, frame in (("validation", validation), ("calibration", calibration)):
        expected = metadata["splits"][split]
        if (frame.time.iloc[0] != expected["start"] or frame.time.iloc[-1] != expected["end"]
                or len(frame) != expected["rows"]):
            raise ValueError("RF/Transformer development split differs")

    targets = [valid_targets(frame.time, length) for frame in train_frames]
    labels = np.concatenate([frame.attack.to_numpy(dtype=np.int64)[indices]
                             for frame, indices in zip(train_frames, targets)])
    chosen = sample_training_indices(labels, config["sampling_seed"], config["max_normal_training_targets"])
    selected_labels = labels[chosen]
    selected_features, selected_rows = [], []
    offset = 0
    for source, frame, indices in zip(("train1", "train2"), train_frames, targets):
        local = chosen[(chosen >= offset) & (chosen < offset+len(indices))] - offset
        picked = indices[local]
        selected_features.append(make_features(transformed(frame, bundle), picked, length))
        selected_rows.append(pd.DataFrame({"source": source, "target_row_index": picked,
                                          "timestamp": frame.time.iloc[picked].to_numpy(),
                                          "label": frame.attack.iloc[picked].to_numpy()}))
        offset += len(indices)
    train_x = np.concatenate(selected_features)
    pd.concat(selected_rows, ignore_index=True).to_csv(artifacts / "training_targets.csv", index=False)
    columns = feature_names(bundle["numeric_columns"] + bundle["binary_columns"])
    validation_targets = valid_targets(validation.time, length)
    calibration_targets = valid_targets(calibration.time, length)
    validation_x = make_features(transformed(validation, bundle), validation_targets, length)
    calibration_x = make_features(transformed(calibration, bundle), calibration_targets, length)
    write_json(artifacts / "training_manifest.json", {
        "candidate_targets": len(labels), "candidate_attack_targets": int(labels.sum()),
        "selected_targets": len(chosen), "selected_attack_targets": int(selected_labels.sum()),
        "selected_normal_targets": int((selected_labels == 0).sum()),
        "feature_count": len(columns), "feature_names": columns,
        "validation_targets": len(validation_targets), "calibration_targets": len(calibration_targets),
        "sources_kept_separate": True,
    })
    print(f"Training: {train_x.shape}, attack targets={selected_labels.sum()}", flush=True)
    # Fit and freeze every seed/threshold before opening test2 for RF scoring.
    for seed in config["seeds"]:
        print(f"Fit Random Forest seed={seed}", flush=True)
        folder = artifacts / f"seed{seed}"
        folder.mkdir()
        started = time.perf_counter()
        baseline = RandomForestBaseline(seed, config["n_estimators"], config["n_jobs"],
                                        config["max_depth"], config["min_samples_leaf"],
                                        config["max_features"], config["class_weight"]).fit(train_x, selected_labels)
        fit_seconds = time.perf_counter()-started
        joblib.dump({"baseline": baseline, "feature_names": columns, "sequence_length": length,
                     "preprocessing_sha256": protocol["preprocessing_sha256"], "seed": seed,
                     "protocol_version": config["protocol_version"]}, folder / "model.joblib")
        restored = joblib.load(folder / "model.joblib")["baseline"]
        np.testing.assert_allclose(restored.score_samples(train_x[:128]), baseline.score_samples(train_x[:128]), atol=0, rtol=0)
        cal_scores = restored.score_samples(calibration_x)
        threshold = calibrate_scores(cal_scores, config["threshold_quantile"])
        threshold.update(score="RandomForest predict_proba Attack; not probability-calibrated",
                         calibration_split="calibration", calibration_windows=len(cal_scores),
                         calibration_fpr=threshold.pop("validation_fpr"),
                         note="Fixed calibration quantile with ties; no test threshold tuning",
                         seed=seed, fit_seconds=fit_seconds, model_sha256=file_hash(folder / "model.joblib"),
                         preprocessing_sha256=protocol["preprocessing_sha256"])
        val_scores = restored.score_samples(validation_x)
        threshold["validation_fpr"] = float(np.mean(val_scores > threshold["threshold"]))
        write_json(folder / "threshold.json", threshold)
        for split, frame, indices, scores in (("validation", validation, validation_targets, val_scores),
                                              ("calibration", calibration, calibration_targets, cal_scores)):
            pd.DataFrame({"timestamp": frame.time.iloc[indices].to_numpy(), "label": frame.attack.iloc[indices].to_numpy(),
                          "score": scores, "prediction": (scores > threshold["threshold"]).astype(int)}).to_csv(folder / f"{split}_scores.csv", index=False)
        pd.DataFrame({"feature": columns, "importance": restored.model.feature_importances_}).sort_values("importance", ascending=False).to_csv(folder / "feature_importances.csv", index=False)
        print(f"seed={seed}, fit={fit_seconds:.1f}s, threshold={threshold['threshold']:.6f}, calibration FPR={threshold['calibration_fpr']:.6f}", flush=True)

    audit = json.loads((ROOT / "manifests/hai_mixed_v2_holdout_audit.json").read_text())
    test_path = ROOT / audit["holdout"]
    if not audit["passed"] or file_hash(test_path) != audit["extracted_sha256"]:
        raise ValueError("Test2 integrity/overlap audit failed")
    test = pd.read_csv(test_path, sep=";")
    validate_test(test, bundle, metadata, length)
    test_targets = valid_targets(test.time, length)
    test_x = make_features(transformed(test, bundle), test_targets, length)
    labels_test = test.attack.iloc[test_targets].to_numpy(dtype=np.int64)
    transformer_dir = ROOT / "results/hai/transformer_mixed/mixed_v2_full_seed42_run1/test_evaluation"
    transformer_predictions = pd.read_csv(transformer_dir / "test_predictions.csv")
    np.testing.assert_array_equal(test.time.iloc[test_targets].to_numpy(), transformer_predictions.timestamp.to_numpy())
    np.testing.assert_array_equal(labels_test, transformer_predictions.label.to_numpy())
    transformer_metrics = json.loads((transformer_dir / "metrics.json").read_text())
    comparisons = []
    for seed in config["seeds"]:
        folder = artifacts / f"seed{seed}"
        output = results / f"seed{seed}"
        output.mkdir()
        saved = joblib.load(folder / "model.joblib")
        threshold = json.loads((folder / "threshold.json").read_text())
        if file_hash(folder / "model.joblib") != threshold["model_sha256"]:
            raise ValueError("RF model/threshold mismatch")
        started = time.perf_counter()
        scores = saved["baseline"].score_samples(test_x)
        prediction_seconds = time.perf_counter()-started
        metrics, predictions = calculate_metrics(labels_test, scores, threshold["threshold"])
        metrics.update(model="Random Forest Classifier", seed=seed, training_access="normal and labelled attacks",
                       test_previously_seen=True, descriptive_comparison=True, tuning_on_test=False,
                       test_sha256=audit["extracted_sha256"], model_sha256=threshold["model_sha256"],
                       fit_seconds=threshold["fit_seconds"], prediction_seconds=prediction_seconds)
        frame = pd.DataFrame({"timestamp": test.time.iloc[test_targets].to_numpy(), "label": labels_test,
                              "score": scores, "prediction": predictions})
        frame.to_csv(output / "test_predictions.csv", index=False)
        write_json(output / "metrics.json", metrics)
        plot_results(frame, metrics, output / "figures")
        comparisons.append(metrics)
        print(f"seed={seed}: F1={metrics['f1']:.4f}, Recall={metrics['recall']:.4f}, FPR={metrics['fpr']:.4f}", flush=True)
    keys = ["precision", "recall", "f1", "fpr", "accuracy", "average_precision"]
    table = [{"model": row["model"], "seed": row["seed"], **{k: row[k] for k in keys}} for row in comparisons]
    table.append({"model": "Transformer mixed v2", "seed": 42, **{k: transformer_metrics[k] for k in keys}})
    pd.DataFrame(table).to_csv(results / "comparison.csv", index=False)
    aggregates = pd.DataFrame([{ "metric": key, "rf_mean": float(np.mean([r[key] for r in comparisons])),
                               "rf_std_population": float(np.std([r[key] for r in comparisons])),
                               "transformer_seed42": transformer_metrics[key]} for key in keys])
    aggregates.to_csv(results / "comparison_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(3)
    means = [np.mean([r[key] for r in comparisons]) for key in ("precision", "recall", "f1")]
    errors = [np.std([r[key] for r in comparisons]) for key in ("precision", "recall", "f1")]
    axes[0].bar(x-.18, means, width=.36, yerr=errors, label="RF mean +/- std, 3 seeds")
    axes[0].bar(x+.18, [transformer_metrics[key] for key in ("precision", "recall", "f1")], width=.36, label="Transformer, seed42")
    axes[0].set(xticks=x, xticklabels=["Precision", "Recall", "F1"], ylim=(0,1.05), title="Same test2 targets; different training access")
    axes[0].legend(fontsize=8)
    axes[1].bar(["RF mean", "Transformer"], [np.mean([r["fpr"] for r in comparisons]), transformer_metrics["fpr"]])
    axes[1].set(ylabel="False positive rate", title="Lower FPR is better")
    fig.tight_layout()
    fig.savefig(results / "comparison.png", dpi=150)
    plt.close(fig)
    lines = ["# Random Forest baseline на HAI", "",
             "RandomForestClassifier обучен на размеченных normal/attack целях из train1 и первых 70% train2. В отличие от него, Transformer v2 обучался прогнозу только на нормальных данных. Это сравнение методов, а не изолированная абляция архитектуры.",
             f"Обучающих целей после train-only downsampling: {len(selected_labels)}, из них атакующих {selected_labels.sum()}. Все атакующие train-цели сохранены; нормальные ограничены 100000 случайно выбранных целей, seed=42. Истории при выборке не разрываются и используют исходные прошлые записи.",
             "Использованы текущие значения, mean/std/min/max за 30 предыдущих секунд и разность с предыдущей записью: 354 признака. Время и все attack-метки исключены из входов. Это компактное представление истории, а не полный flattened Transformer-вход.",
             "Scaler v2 не переобучался. Гиперпараметры, sampling и правило порога зафиксированы до RF test-scoring, перебор по test не выполнялся. RF-порог — 99-й процентиль score на отдельной нормальной calibration; из-за дискретных score/ties FPR может отличаться от 1%.",
             "", "| Модель | Seed | Precision | Recall | F1 | FPR | AP |", "|---|---|---|---|---|---|---|"]
    for row in table:
        lines.append(f"| {row['model']} | {row['seed']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['fpr']:.4f} | {row['average_precision']:.4f} |")
    lines += ["", "Test2 содержит ровно те же 152970 оцениваемых целей и метки, что ранее сохранённая оценка Transformer. Первый warmup 30 строк исключён. Метрики поточечные, без point adjustment. RF представлен всеми тремя seeds без выбора лучшего по test; Transformer — одним checkpoint прерванного после 22 эпох запуска.",
              "Test2 уже был просмотрен при разработке предыдущего этапа. Это ретроспективное описательное сравнение на известном benchmark, не новый слепой holdout. Различие в разметке и объёме train, sampling и представлении истории не позволяет приписывать разницу только архитектуре. Обучающие атаки относятся лишь к двум размеченным интервалам train2; результат не доказывает zero-day обобщение.",
              "Время пакетного predict_proba не является latency batch=1. Сохраняются модели joblib, пороги, train target manifest, feature importances, метрики, предсказания и графики. Воспроизведение: python -m src.run_random_forest_hai (существующая версия не перезаписывается)."]
    (results / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    write_json(results / "status.json", {"status": "completed", "seeds": config["seeds"],
                                         "test_targets": len(test_targets), "targets_match_transformer": True})
    print("Saved comparison:", results, flush=True)


if __name__ == "__main__":
    main()
