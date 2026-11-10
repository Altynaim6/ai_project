import json

import joblib
import numpy as np
import pandas as pd
import pytest
import torch

from src.config import PROJECT_ROOT
from src.mixed_window_dataset import MixedForecastWindowDataset
from src.prepare_hai_v2 import attack_exclusion_mask, prepare_v2
from src.range_floor_scaler import RangeFloorScaler


def test_floor_controls_low_variance_without_clipping_or_refit():
    train = pd.DataFrame({"valve": [12., 12.001, 12.002], "speed": [20., 20., 20.]})
    scaler = RangeFloorScaler({"valve": [0, 100], "speed": [0, 3200]}).fit(train)
    np.testing.assert_allclose(scaler.scale_, [1., 32.])
    mean = scaler.mean_.copy()
    test = pd.DataFrame({"valve": [200.], "speed": [0.]})
    transformed = scaler.transform(test)
    assert transformed[0, 0] > 100  # no clipping, including beyond engineering range
    assert transformed[0, 1] == pytest.approx(-.625)
    np.testing.assert_array_equal(scaler.mean_, mean)
    np.testing.assert_allclose(scaler.inverse_transform(transformed), test)
    with pytest.raises(ValueError, match="order"):
        scaler.transform(test[["speed", "valve"]])


def test_all_attack_flags_and_guard_are_used():
    frame = pd.DataFrame({c: np.zeros(10) for c in ("attack", "attack_P1", "attack_P2", "attack_P3")})
    frame.loc[5, "attack_P2"] = 1
    attacks, excluded = attack_exclusion_mask(frame, guard_seconds=1)
    assert np.flatnonzero(attacks).tolist() == [5]
    assert np.flatnonzero(excluded).tolist() == [4, 5, 6]


def test_windows_do_not_bridge_excluded_intervals(tmp_path):
    times = np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01",
                      "2020-01-01T00:00:02", "2020-01-01T00:00:03",
                      "2020-01-01T00:01:00", "2020-01-01T00:01:01",
                      "2020-01-01T00:01:02", "2020-01-01T00:01:03"])
    path = tmp_path / "segments.npz"
    np.savez(path, numeric=np.arange(8, dtype=np.float32).reshape(-1, 1),
             binary=np.zeros((8, 1), dtype=np.int64), timestamps=times,
             labels=np.zeros(8), numeric_columns=np.array(["sensor"]),
             binary_columns=np.array(["state"]), schema_sha256=np.array("test"))
    dataset = MixedForecastWindowDataset(path, sequence_length=2)
    assert dataset.segment_count == 2
    assert dataset.target_indices.tolist() == [2, 3, 6, 7]
    history, target, _ = dataset[2]
    torch.testing.assert_close(history[:, 0], torch.tensor([4., 5.]))
    assert target.item() == 6.


def test_v2_split_fit_and_reserved_test_are_independent(tmp_path):
    (tmp_path / "configs").mkdir()
    schema = json.loads((PROJECT_ROOT / "configs/hai_20_07_schema.json").read_text())
    (tmp_path / "configs/hai_20_07_schema.json").write_text(json.dumps(schema))
    raw = tmp_path / "data/raw/hai"
    raw.mkdir(parents=True)
    for name, count, date in (("train1", 500, "2020-01-01"), ("train2", 4000, "2020-01-02")):
        data = {"time": pd.date_range(date, periods=count, freq="s").astype(str)}
        for channel in schema["feature_order"]:
            data[channel] = np.zeros(count) if channel in schema["binary_columns"] else np.arange(count, dtype=float)
        for channel in ("attack", "attack_P1", "attack_P2", "attack_P3"):
            data[channel] = np.zeros(count, dtype=int)
        frame = pd.DataFrame(data)
        if name == "train2":
            frame.loc[1000:1004, "attack_P1"] = 1
            frame.loc[2000:2004, "attack"] = 1
            frame.loc[3400:, schema["numeric_columns"]] = 1e6  # calibration shift must not affect fit
        frame.to_csv(raw / f"{name}.csv", sep=";", index=False)
    # No test files exist. Preparation must complete without reading either test.
    manifest = prepare_v2(tmp_path)
    assert manifest["source_audit"]["train2"]["attack_flagged_rows"] == 10
    assert manifest["source_audit"]["train2"]["excluded_with_guard"] == 1210
    assert manifest["splits"]["train"]["rows"] == 2090
    assert manifest["splits"]["validation"]["rows"] == 600
    assert manifest["splits"]["calibration"]["rows"] == 600
    bundle = joblib.load(tmp_path / "artifacts/hai/preprocessing_hai_mixed_v2/preprocessing.joblib")
    # Training indices: train1 0:500; train2 0:700,1305:1700,2305:2800.
    expected = np.concatenate([np.arange(500), np.arange(700), np.arange(1305, 1700), np.arange(2305, 2800)])
    np.testing.assert_allclose(bundle["scaler"].mean_, np.full(55, expected.mean()))
    train = MixedForecastWindowDataset(tmp_path / "data/processed/hai_mixed_v2/train.npz")
    assert train.segment_count == 4
    assert len(train) == 2090 - 4 * 30
    assert np.all(train.labels == 0)
    protocol = manifest["protocol"]
    assert not protocol["fresh_test"]["opened_by_preprocessing"]
    assert protocol["fresh_test"]["status"] == "reserved_uninspected"
    with pytest.raises(FileExistsError):
        prepare_v2(tmp_path)
