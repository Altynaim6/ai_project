import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import PROJECT_ROOT
from src.preprocess_hai_mixed import prepare
from src.train_transformer_mixed import calibrate_scores, run_epoch, run_training
from src.evaluate_transformer_mixed import calculate_metrics, evaluate, load_frozen_run


class ConstantForecast(torch.nn.Module):
    def forward(self, history):
        return torch.zeros(len(history), 1), torch.zeros(len(history), 1, 2)


def test_full_validation_includes_partial_batch_and_weighted_mean():
    numeric = torch.tensor([[0.], [1.], [2.], [3.], [4.]])
    loader = DataLoader(TensorDataset(torch.zeros(5, 2, 2), numeric,
                                     torch.zeros(5, 1, dtype=torch.long)), batch_size=2)
    metrics, scores = run_epoch(ConstantForecast(), loader, torch.device("cpu"),
                                collect_scores=True, progress_every=0)
    assert metrics["windows"] == 5
    assert metrics["batches"] == 3
    np.testing.assert_allclose(scores, np.arange(5) ** 2 + np.log(2), rtol=1e-6)
    assert metrics["loss"] == pytest.approx(float(scores.mean()), rel=1e-6)
    assert metrics["numeric_mse"] == pytest.approx(6.)
    assert metrics["binary_ce"] == pytest.approx(np.log(2), rel=1e-6)
    with pytest.raises(ValueError, match="only for training"):
        run_epoch(ConstantForecast(), loader, "cpu", max_train_batches=1)


def test_calibration_strict_rule_and_ties():
    result = calibrate_scores(np.array([0., 1., 2., 3., 4.]), quantile=0.5)
    assert result["threshold"] == 2.
    assert result["validation_fpr"] == 0.4
    assert calibrate_scores(np.ones(10))["validation_fpr"] == 0.
    with pytest.raises(ValueError):
        calibrate_scores([np.nan])


def prepared_fixture(tmp_path):
    schema_path = PROJECT_ROOT / "configs/hai_20_07_schema.json"
    schema = json.loads(schema_path.read_text())
    for split, offset in (("train", 0), ("validation", 40)):
        data = {"time": pd.date_range("2020-01-01", periods=80, freq="s")[offset:offset+40].astype(str)}
        for name in schema["feature_order"]:
            data[name] = (np.arange(40) % 2 if name in schema["binary_columns"]
                          else np.arange(40, dtype=float))
        for name in ("attack", "attack_P1", "attack_P2", "attack_P3"):
            data[name] = np.zeros(40, dtype=int)
        pd.DataFrame(data).to_csv(tmp_path / f"{split}.csv", sep=";", index=False)
    prepare(tmp_path / "train.csv", tmp_path / "validation.csv", schema_path,
            tmp_path / "data", tmp_path / "preprocessing")


def test_early_stopping_restores_best_and_calibrates_full_split(tmp_path, monkeypatch):
    prepared_fixture(tmp_path)
    state = {"validation_calls": 0, "training_calls": 0, "best_parameter": None}

    def fake_epoch(model, loader, device, optimizer=None, max_train_batches=None,
                   collect_scores=False, progress_every=100):
        if optimizer is not None:
            state["training_calls"] += 1
            with torch.no_grad():
                next(model.parameters()).add_(1)
            loss = 1.
        elif not collect_scores:
            state["validation_calls"] += 1
            loss = float(state["validation_calls"])
            if state["validation_calls"] == 1:
                state["best_parameter"] = next(model.parameters()).detach().clone()
        else:
            torch.testing.assert_close(next(model.parameters()), state["best_parameter"])
            loss = 1.
        metrics = {"loss": loss, "numeric_mse": loss, "binary_ce": 0.,
                   "windows": len(loader.dataset), "batches": len(loader), "seconds": 0.}
        return metrics, np.ones(len(loader.dataset)) if collect_scores else None

    monkeypatch.setattr("src.train_transformer_mixed.run_epoch", fake_epoch)
    args = (tmp_path / "data", tmp_path / "preprocessing",
            tmp_path / "model", tmp_path / "results",
            dict(max_epochs=5, early_stopping_patience=1, batch_size=8, learning_rate=.001),
            dict(num_numeric=55, num_binary=4, sequence_length=4,
                 d_model=8, nhead=2, num_layers=1, dropout=0.))
    summary = run_training(*args, progress_every=0)
    assert summary["epochs_completed"] == 2
    assert summary["best_epoch"] == 1
    assert summary["stopped_early"]
    assert summary["validation_windows"] == 36
    assert summary["threshold"] == 1.
    checkpoint = torch.load(tmp_path / "model/best_model.pt", weights_only=True)
    assert checkpoint["epoch"] == 1
    assert len(pd.read_csv(tmp_path / "results/validation_scores.csv")) == 36
    assert (tmp_path / "model/preprocessing.joblib").exists()
    with pytest.raises(FileExistsError):
        run_training(*args, progress_every=0)


def test_test_metrics_known_confusion_and_absent_class():
    metrics, predictions = calculate_metrics([0, 0, 1, 1], [0., 1., 2., 3.], 1.)
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (2, 0, 0, 2)
    assert metrics["f1"] == 1.
    assert predictions.tolist() == [0, 0, 1, 1]
    metrics, _ = calculate_metrics([0, 0], [0., 0.], 1.)
    assert metrics["recall"] is None
    assert metrics["average_precision"] is None
    assert metrics["precision"] == 0.


def test_smoke_run_rejected_before_test_read(tmp_path):
    (tmp_path / "threshold.json").write_text(json.dumps({"run_type": "smoke_test"}))
    (tmp_path / "run_config.json").write_text(json.dumps({"run_type": "smoke_test"}))
    with pytest.raises(ValueError, match="smoke-test"):
        load_frozen_run(tmp_path)


def test_evaluation_end_to_end_synthetic_frozen_threshold_and_plots(tmp_path):
    prepared_fixture(tmp_path)
    torch.set_num_threads(2)
    run_training(
        tmp_path / "data", tmp_path / "preprocessing", tmp_path / "model", tmp_path / "results",
        dict(max_epochs=1, early_stopping_patience=1, batch_size=8,
             validation_batch_size=7, learning_rate=.001),
        dict(num_numeric=55, num_binary=4, sequence_length=4,
             d_model=8, nhead=2, num_layers=1, dropout=0.), progress_every=0,
    )
    threshold_before = (tmp_path / "model/threshold.json").read_bytes()
    test = pd.read_csv(tmp_path / "validation.csv", sep=";")
    test["time"] = (pd.to_datetime(test["time"]) + pd.Timedelta(days=1)).astype(str)
    test.loc[10:19, ["attack", "attack_P1"]] = 1
    test.to_csv(tmp_path / "test.csv", sep=";", index=False)
    metrics = evaluate(tmp_path / "model", tmp_path / "test.csv", tmp_path / "evaluation", batch_size=7)
    assert metrics["source_rows"] == 40
    assert metrics["evaluated_windows"] == 36
    assert metrics["attack_windows"] == 10
    assert metrics["normal_windows"] == 26
    assert (tmp_path / "model/threshold.json").read_bytes() == threshold_before
    predictions = pd.read_csv(tmp_path / "evaluation/test_predictions.csv")
    assert predictions["prediction"].tolist() == (predictions["score"] > metrics["threshold"]).astype(int).tolist()
    assert len(list((tmp_path / "evaluation/figures").glob("*.png"))) == 4
    assert json.loads((tmp_path / "evaluation/evaluation_status.json").read_text())["status"] == "completed"
    # A changed threshold/model association must not silently pass.
    calibration = json.loads(threshold_before)
    calibration["model_sha256"] = "incorrect"
    (tmp_path / "model/threshold.json").write_text(json.dumps(calibration))
    with pytest.raises(ValueError, match="Hash mismatch"):
        load_frozen_run(tmp_path / "model")


def test_separate_calibration_defines_threshold_not_validation(tmp_path):
    prepared_fixture(tmp_path)
    with np.load(tmp_path / "data/validation.npz") as data:
        values = {name: data[name].copy() for name in data.files}
    values["numeric"] = values["numeric"][:20] + 100
    for name in ("binary", "labels"):
        values[name] = values[name][:20]
    values["timestamps"] = pd.date_range("2020-01-03", periods=20, freq="s").astype(str).to_numpy(dtype=str)
    np.savez(tmp_path / "data/calibration.npz", **values)
    torch.set_num_threads(2)
    summary = run_training(
        tmp_path / "data", tmp_path / "preprocessing", tmp_path / "model", tmp_path / "results",
        dict(max_epochs=1, early_stopping_patience=1, batch_size=8,
             validation_batch_size=7, learning_rate=.001),
        dict(num_numeric=55, num_binary=4, sequence_length=4,
             d_model=8, nhead=2, num_layers=1, dropout=0.), progress_every=0,
    )
    assert summary["calibration_split"] == "calibration"
    assert summary["validation_windows"] == 36
    assert summary["calibration_windows"] == 16
    calibration = pd.read_csv(tmp_path / "results/calibration_scores.csv")
    assert summary["threshold"] == pytest.approx(float(np.quantile(calibration.score, .99)), rel=1e-6)
    assert summary["validation_fpr"] == 0.
    assert summary["calibration_fpr"] > 0.
    saved = json.loads((tmp_path / "model/threshold.json").read_text())
    assert saved["validation_windows"] == 36
    assert saved["calibration_windows"] == 16
