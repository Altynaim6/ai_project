import json

import joblib
import numpy as np
import pandas as pd
import pytest
import torch

from src.config import PROJECT_ROOT
from src.mixed_forecast_loss import mixed_forecast_scores
from src.mixed_window_dataset import MixedForecastWindowDataset
from src.models.transformer_mixed import MixedForecastTransformer
from src.preprocess_hai_mixed import prepare, transform_frame


def fixture_frames():
    schema = json.loads((PROJECT_ROOT / "configs/hai_20_07_schema.json").read_text())
    count = 40
    frames = []
    for offset in (0, count):
        data = {"time": pd.date_range("2020-01-01", periods=80, freq="s")[offset:offset+count].astype(str)}
        for name in schema["feature_order"]:
            data[name] = (np.arange(count) % 2 if name in schema["binary_columns"]
                          else np.arange(count, dtype=float) + offset * 100)
        for name in ("attack", "attack_P1", "attack_P2", "attack_P3"):
            data[name] = np.zeros(count, dtype=int)
        frames.append(pd.DataFrame(data))
    return schema, frames


def test_train_only_fit_binary_preservation_and_window_target(tmp_path):
    schema, (train, validation) = fixture_frames()
    train.to_csv(tmp_path / "train.csv", sep=";", index=False)
    validation.to_csv(tmp_path / "validation.csv", sep=";", index=False)
    schema_path = PROJECT_ROOT / "configs/hai_20_07_schema.json"
    bundle = prepare(tmp_path / "train.csv", tmp_path / "validation.csv",
                     schema_path, tmp_path / "data", tmp_path / "artifacts")
    np.testing.assert_allclose(bundle["scaler"].mean_, np.full(55, 19.5))
    restored = joblib.load(tmp_path / "artifacts/preprocessing.joblib")
    invalid = validation.copy()
    invalid["P2_Auto"] = 0.5
    with pytest.raises(ValueError, match="Invalid transformed"):
        transform_frame(invalid, restored)
    expected_numeric, expected_binary = transform_frame(validation, restored)
    with np.load(tmp_path / "data/validation.npz") as data:
        np.testing.assert_array_equal(data["binary"], expected_binary)
        np.testing.assert_allclose(data["numeric"], expected_numeric)
    dataset = MixedForecastWindowDataset(tmp_path / "data/train.npz", 30)
    history, numeric, binary = dataset[0]
    assert len(dataset) == 10
    np.testing.assert_array_equal(history[:, 55:].numpy(), train[schema["binary_columns"]].iloc[:30])
    expected_train, _ = transform_frame(train, bundle)
    np.testing.assert_allclose(numeric.numpy(), expected_train[30])
    np.testing.assert_array_equal(binary.numpy(), train[schema["binary_columns"]].iloc[30])
    assert binary.dtype == torch.long
    # A future input value cannot enter this window or its target.
    original = history.clone()
    dataset.numeric[31] = 999
    torch.testing.assert_close(dataset[0][0], original)
    with pytest.raises(IndexError):
        dataset[len(dataset)]


def test_preparation_rejects_invalid_binary_and_overlap(tmp_path):
    _, (train, validation) = fixture_frames()
    train.loc[0, "P2_Auto"] = 2
    train.to_csv(tmp_path / "train.csv", sep=";", index=False)
    validation.to_csv(tmp_path / "validation.csv", sep=";", index=False)
    args = (tmp_path / "train.csv", tmp_path / "validation.csv",
            PROJECT_ROOT / "configs/hai_20_07_schema.json", tmp_path / "data", tmp_path / "artifacts")
    with pytest.raises(ValueError, match="Binary states"):
        prepare(*args)
    train.loc[0, "P2_Auto"] = 0
    train.to_csv(tmp_path / "train.csv", sep=";", index=False)
    train.to_csv(tmp_path / "validation.csv", sep=";", index=False)
    with pytest.raises(ValueError, match="overlap"):
        prepare(*args)


def test_loss_known_values_and_binary_target_validation():
    scores = mixed_forecast_scores(torch.ones(2, 55), torch.zeros(2, 4, 2),
                                   torch.zeros(2, 55), torch.zeros(2, 4, dtype=torch.long))
    torch.testing.assert_close(scores, torch.full((2,), 1 + np.log(2), dtype=torch.float32))
    with pytest.raises(ValueError, match="0 or 1"):
        mixed_forecast_scores(torch.ones(2, 55), torch.zeros(2, 4, 2),
                              torch.zeros(2, 55), torch.full((2, 4), 2))


def test_causal_mask_and_both_heads_gradients():
    torch.manual_seed(42)
    model = MixedForecastTransformer(sequence_length=8, d_model=16, nhead=4, dropout=0)
    model.eval()
    history = torch.randn(2, 8, 59)
    changed = history.clone()
    changed[:, 4:] += 100
    with torch.no_grad():
        torch.testing.assert_close(model.encode(history)[:, :4], model.encode(changed)[:, :4], rtol=0, atol=1e-5)
    prediction, logits = model(history)
    assert prediction.shape == (2, 55)
    assert logits.shape == (2, 4, 2)
    scores = mixed_forecast_scores(prediction, logits, torch.zeros(2, 55), torch.zeros(2, 4, dtype=torch.long))
    scores.mean().backward()
    for head in (model.numeric_head, model.binary_head):
        assert head.weight.grad is not None
        assert torch.isfinite(head.weight.grad).all()
        assert head.weight.grad.abs().sum() > 0
