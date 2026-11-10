import joblib
import numpy as np
import pytest

from src.models.baseline import (RandomForestBaseline, feature_names, make_features,
                                 sample_training_indices, valid_targets)


def test_causal_features_known_values_and_future_invariance():
    values = np.arange(8, dtype=np.float32).reshape(-1, 1)
    features = make_features(values, np.array([3]), sequence_length=3)
    np.testing.assert_allclose(features[0], [3., 1., np.sqrt(2/3), 0., 2., 1.], atol=1e-6)
    values[4:] = 999
    np.testing.assert_array_equal(make_features(values, np.array([3]), 3), features)
    assert len(feature_names(["sensor", "state"])) == 12
    with pytest.raises(ValueError):
        make_features(values, np.array([1]), 3)


def test_targets_do_not_cross_gaps_or_split_boundaries():
    times = np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01",
                      "2020-01-01T00:00:02", "2020-01-01T00:00:03",
                      "2020-01-01T01:00:00", "2020-01-01T01:00:01",
                      "2020-01-01T01:00:02", "2020-01-01T01:00:03"])
    assert valid_targets(times, 2).tolist() == [2, 3, 6, 7]
    assert valid_targets(times[:4], 2).tolist() == [2, 3]
    assert valid_targets(times[4:], 2).tolist() == [2, 3]
    with pytest.raises(ValueError):
        valid_targets(np.array([times[0], times[0]]))


def test_train_only_sampling_retains_all_attacks_and_is_reproducible():
    labels = np.array([0]*100 + [1]*7)
    indices = sample_training_indices(labels, seed=42, max_normal=10)
    np.testing.assert_array_equal(indices, sample_training_indices(labels, 42, 10))
    assert len(indices) == 17 and labels[indices].sum() == 7
    np.testing.assert_array_equal(indices[-7:], np.arange(100, 107))
    with pytest.raises(ValueError):
        sample_training_indices(np.zeros(100))


def test_rf_learns_two_classes_and_saved_scores_match(tmp_path):
    x = np.r_[np.zeros((20, 6)), np.ones((20, 6))].astype(np.float32)
    y = np.r_[np.zeros(20), np.ones(20)].astype(int)
    baseline = RandomForestBaseline(n_estimators=10, n_jobs=1).fit(x, y)
    scores = baseline.score_samples(x)
    assert scores[:20].max() < .5 and scores[20:].min() > .5
    path = tmp_path / "model.joblib"
    joblib.dump(baseline, path)
    np.testing.assert_array_equal(joblib.load(path).score_samples(x), scores)
    with pytest.raises(ValueError, match="normal and attack"):
        RandomForestBaseline().fit(x, np.zeros(40))
