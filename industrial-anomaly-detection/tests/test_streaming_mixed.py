import json
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import PROJECT_ROOT
from src.range_floor_scaler import RangeFloorScaler
from src.simulate_transformer_mixed import StreamingDetector, simulate


class LastValueModel(torch.nn.Module):
    def forward(self, history):
        return history[:, -1, :1], torch.zeros(len(history), 1, 2)


def toy_detector(length=2, threshold=2.):
    scaler = RangeFloorScaler({"sensor": [0, 100]}).fit(pd.DataFrame({"sensor": [0., 0.]}))
    return StreamingDetector(LastValueModel(), {"scaler": scaler, "numeric_columns": ["sensor"],
                                               "binary_columns": ["state"]}, threshold, length)


def row(second, value=0):
    return {"time": f"2020-01-01 00:00:{second:02d}", "sensor": str(value), "state": "0", "attack": "999"}


def test_warmup_target_shift_threshold_and_labels_ignored():
    detector = toy_detector(threshold=1 + np.log(2))
    assert detector.push(row(0, 0))["status"] == "WARMUP"
    assert detector.push(row(1, 1))["status"] == "WARMUP"
    result = detector.push(row(2, 2))
    assert result["score"] == pytest.approx(1 + np.log(2), abs=1e-6)
    assert result["history_size"] == 2
    # Exact saved rule is strictly >, not >=.
    detector = toy_detector(threshold=result["score"])
    detector.push(row(0, 0))
    detector.push(row(1, 1))
    assert detector.push(row(2, 2))["status"] == "NORMAL"
    changed_label = row(3, 4)
    changed_label["attack"] = "0"
    assert detector.push(changed_label)["status"] == "ANOMALY"


def test_gap_invalid_and_out_of_order_restart_history():
    detector = toy_detector()
    detector.push(row(0))
    detector.push(row(1))
    gap = detector.push(row(10))
    assert gap["status"] == "WARMUP" and "reset" in gap["detail"]
    assert detector.push(row(10))["status"] == "INSUFFICIENT_DATA"
    assert len(detector.buffer) == 0
    bad = row(11)
    bad["sensor"] = "nan"
    assert detector.push(bad)["status"] == "INSUFFICIENT_DATA"
    bad = row(12)
    bad["state"] = "0.5"
    assert detector.push(bad)["status"] == "INSUFFICIENT_DATA"
    assert detector.push(row(13))["status"] == "WARMUP"
    assert detector.push(row(14))["status"] == "WARMUP"
    assert detector.push(row(15))["status"] == "NORMAL"


def test_replay_logs_each_prediction_and_each_alert(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "best_model.pt").write_bytes(b"synthetic fixture")
    (model_dir / "threshold.json").write_text("{}")
    monkeypatch.setattr(StreamingDetector, "from_run", lambda path: toy_detector(threshold=0.))
    source = tmp_path / "source.csv"
    with source.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row(0)), delimiter=";")
        writer.writeheader()
        writer.writerows([row(i, i) for i in range(6)])
    output = tmp_path / "output"
    summary = simulate(model_dir, source, output, max_rows=5, quiet=True)
    history = pd.read_csv(output / "predictions.csv")
    alerts = pd.read_csv(output / "alerts.csv")
    assert summary["counts"] == {"WARMUP": 2, "NORMAL": 0, "ANOMALY": 3, "INSUFFICIENT_DATA": 0}
    assert len(history) == 5 and len(alerts) == 3
    assert not {"attack", "label"} & set(history.columns)
    assert alerts.event_id.tolist() == ["row-2", "row-3", "row-4"]
    assert json.loads((output / "run_config.json").read_text())["ground_truth_used"] is False
    with pytest.raises(FileExistsError):
        simulate(model_dir, source, output)


def test_real_saved_model_streaming_matches_offline_scores():
    model_dir = PROJECT_ROOT / "artifacts/hai/transformer_mixed/mixed_v2_full_seed42_run1"
    offline_file = PROJECT_ROOT / "results/hai/transformer_mixed/mixed_v2_full_seed42_run1/test_evaluation/test_predictions.csv"
    if not model_dir.exists() or not offline_file.exists():
        pytest.skip("Local trained artifacts not available")
    torch.set_num_threads(2)
    detector = StreamingDetector.from_run(model_dir)
    frame = pd.read_csv(PROJECT_ROOT / "data/raw/hai/holdout/test2.csv", sep=";", nrows=330)
    scores, predictions = [], []
    for observation in frame.to_dict("records"):
        result = detector.push(observation)
        if result["prediction"] is not None:
            scores.append(result["score"])
            predictions.append(result["prediction"])
    expected = pd.read_csv(offline_file, nrows=300)
    np.testing.assert_allclose(scores, expected.score, atol=1e-5, rtol=1e-4)
    np.testing.assert_array_equal(predictions, expected.prediction)
