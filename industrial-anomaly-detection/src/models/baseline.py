"""Random Forest classifier using current values and causal history summaries."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier


GROUPS = ("current", "history_mean", "history_std", "history_min", "history_max", "delta")


def feature_names(channels):
    return [f"{group}__{channel}" for group in GROUPS for channel in channels]


def valid_targets(timestamps, sequence_length=30):
    if sequence_length < 1:
        raise ValueError("History length must be positive")
    times = np.asarray(timestamps, dtype="datetime64[s]")
    if np.isnat(times).any() or np.any(np.diff(times) <= np.timedelta64(0, "s")):
        raise ValueError("Invalid/duplicate/out-of-order timestamps")
    boundaries = np.r_[0, np.flatnonzero(np.diff(times) != np.timedelta64(1, "s")) + 1, len(times)]
    parts = [np.arange(start + sequence_length, end, dtype=np.int64)
             for start, end in zip(boundaries[:-1], boundaries[1:]) if end-start > sequence_length]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def make_features(values, targets, sequence_length=30, batch_size=2048):
    """One row per target t, based only on x[t-L:t+1]. No labels/time IDs."""
    values = np.asarray(values, dtype=np.float32)
    targets = np.asarray(targets)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite feature matrix")
    if (sequence_length < 1 or batch_size < 1 or targets.ndim != 1
            or not np.issubdtype(targets.dtype, np.integer)
            or np.any(targets < sequence_length) or np.any(targets >= len(values))):
        raise ValueError("Invalid target indices/history length")
    result = np.empty((len(targets), len(GROUPS)*values.shape[1]), dtype=np.float32)
    offsets = np.arange(sequence_length, 0, -1)
    for start in range(0, len(targets), batch_size):
        selected = targets[start:start+batch_size]
        history = values[selected[:, None] - offsets[None, :]]
        current = values[selected]
        result[start:start+len(selected)] = np.concatenate([
            current, history.mean(axis=1), history.std(axis=1),
            history.min(axis=1), history.max(axis=1), current-values[selected-1],
        ], axis=1)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite engineered features")
    return result


class RandomForestBaseline:
    def __init__(self, seed=42, n_estimators=100, n_jobs=4, max_depth=16,
                 min_samples_leaf=2, max_features="sqrt", class_weight="balanced_subsample"):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            max_features=max_features, class_weight=class_weight,
            random_state=seed, n_jobs=n_jobs,
        )

    def fit(self, features, labels):
        labels = np.asarray(labels)
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("Random Forest Classifier requires normal and attack training examples")
        self.model.fit(features, labels)
        return self

    def score_samples(self, features):
        positive = np.flatnonzero(self.model.classes_ == 1)
        if len(positive) != 1:
            raise ValueError("Attack class missing from fitted model")
        return self.model.predict_proba(features)[:, positive[0]]


def sample_training_indices(labels, seed=42, max_normal=100000):
    labels = np.asarray(labels)
    if not np.isin(labels, [0, 1]).all() or max_normal < 1:
        raise ValueError("Invalid labels or sampling limit")
    normal = np.flatnonzero(labels == 0)
    attack = np.flatnonzero(labels == 1)
    if not len(normal) or not len(attack):
        raise ValueError("Both training classes are required")
    if len(normal) > max_normal:
        normal = np.random.default_rng(seed).choice(normal, max_normal, replace=False)
    return np.sort(np.r_[normal, attack])
