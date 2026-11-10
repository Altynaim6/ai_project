"""Calibrate and evaluate the best saved interrupted v2 run, without retraining."""
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.config import PROJECT_ROOT as ROOT
from src.models.transformer_mixed import MixedForecastTransformer
from src.mixed_window_dataset import MixedForecastWindowDataset
from src.train_transformer_mixed import file_hash, run_epoch, calibrate_scores, write_json
from src.evaluate_transformer_mixed import evaluate


def main():
    torch.set_num_threads(4)
    name = "mixed_v2_full_seed42_run1"
    model_dir = ROOT / "artifacts/hai/transformer_mixed" / name
    results = ROOT / "results/hai/transformer_mixed" / name
    config = json.loads((model_dir / "run_config.json").read_text())
    checkpoint = torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=True)
    history = pd.read_csv(results / "history.csv")
    best = history.loc[history.validation_loss.idxmin()]
    if checkpoint["epoch"] != int(best.epoch):
        raise ValueError("Checkpoint is not the best recorded epoch")
    if file_hash(model_dir / "preprocessing.joblib") != checkpoint["preprocessing_sha256"]:
        raise ValueError("Preprocessing changed")
    model = MixedForecastTransformer(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    outputs = {}
    for split in ("validation", "calibration"):
        path = ROOT / f"data/processed/hai_mixed_v2/{split}.npz"
        if file_hash(path) != config["data_sha256"][split]:
            raise ValueError("Prepared split changed")
        dataset = MixedForecastWindowDataset(path, checkpoint["model_config"]["sequence_length"])
        if not np.all(dataset.labels == 0):
            raise ValueError("Calibration/validation must be normal")
        metrics, scores = run_epoch(model, DataLoader(dataset, batch_size=256), "cpu", collect_scores=True)
        outputs[split] = (dataset, metrics, scores)
    if not np.isclose(outputs["validation"][1]["loss"], checkpoint["validation_loss"], rtol=1e-5, atol=1e-6):
        raise ValueError("Restored validation loss differs")
    threshold = calibrate_scores(outputs["calibration"][2])
    threshold.update(
        calibration_split="calibration", calibration_windows=len(outputs["calibration"][2]),
        calibration_fpr=threshold["validation_fpr"],
        validation_windows=len(outputs["validation"][2]),
        validation_fpr=float(np.mean(outputs["validation"][2] > threshold["threshold"])),
        model_sha256=file_hash(model_dir / "best_model.pt"),
        preprocessing_sha256=checkpoint["preprocessing_sha256"],
        validation_data_sha256=config["data_sha256"]["validation"],
        calibration_data_sha256=config["data_sha256"]["calibration"],
        best_epoch=checkpoint["epoch"], run_type="full_training",
        training_status="interrupted_after_22_completed_epochs",
        note="Best recorded checkpoint; threshold fitted on separate normal calibration. No test tuning.",
    )
    if (model_dir / "threshold.json").exists():
        raise FileExistsError("Threshold already exists; do not recalibrate silently")
    write_json(model_dir / "threshold.json", threshold)
    for split, (dataset, metrics, scores) in outputs.items():
        pd.DataFrame({"timestamp": dataset.timestamps[dataset.target_indices],
                      "label": dataset.labels[dataset.target_indices], "score": scores,
                      "prediction": (scores > threshold["threshold"]).astype(int)}).to_csv(results / f"{split}_scores.csv", index=False)
    write_json(results / "recovery_summary.json", {
        "training_status": "interrupted", "epochs_completed": int(history.epoch.max()),
        "best_epoch": checkpoint["epoch"], "best_validation_loss": checkpoint["validation_loss"],
        "early_stopping_completed": False, "threshold": threshold["threshold"],
        "calibration_fpr": threshold["calibration_fpr"], "validation_fpr": threshold["validation_fpr"],
    })
    audit = json.loads((ROOT / "manifests/hai_mixed_v2_holdout_audit.json").read_text())
    test_path = ROOT / audit["holdout"]
    if not audit["passed"] or file_hash(test_path) != audit["extracted_sha256"]:
        raise ValueError("Holdout identity/overlap audit failed")
    evaluate(model_dir, test_path, results / "test_evaluation")


if __name__ == "__main__":
    main()
