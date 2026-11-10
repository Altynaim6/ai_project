"""Train-fitted scaling with a documented engineering-range scale floor."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class RangeFloorScaler(TransformerMixin, BaseEstimator):
    """(x - train mean) / max(train std, fraction * engineering range).

    No clipping and no per-test fitting. Constant channels remain observable.
    engineering_ranges maps column names to (documented min, documented max).
    """
    def __init__(self, engineering_ranges, floor_fraction=0.01):
        self.engineering_ranges = engineering_ranges
        self.floor_fraction = floor_fraction

    def fit(self, X, y=None):
        if not 0 < self.floor_fraction <= 1:
            raise ValueError("floor_fraction must be in (0, 1]")
        if not hasattr(X, "columns") or X.empty:
            raise ValueError("Fit requires a non-empty named DataFrame")
        self.feature_names_in_ = np.array(X.columns, dtype=str)
        self.n_features_in_ = len(self.feature_names_in_)
        values = X.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite numeric data")
        widths = np.array([
            self.engineering_ranges[name][1] - self.engineering_ranges[name][0]
            for name in self.feature_names_in_
        ], dtype=np.float64)
        if not np.isfinite(widths).all() or not (widths > 0).all():
            raise ValueError("Engineering ranges must have finite positive widths")
        self.mean_ = values.mean(axis=0)
        self.var_ = values.var(axis=0)
        self.scale_floor_ = widths * self.floor_fraction
        self.scale_ = np.maximum(np.sqrt(self.var_), self.scale_floor_)
        self.floor_applied_ = np.sqrt(self.var_) < self.scale_floor_
        return self

    def transform(self, X):
        check_is_fitted(self, "scale_")
        if hasattr(X, "columns") and list(X.columns) != self.feature_names_in_.tolist():
            raise ValueError("Numeric feature order differs from fitted order")
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_ or not np.isfinite(values).all():
            raise ValueError("Invalid numeric input")
        return (values - self.mean_) / self.scale_

    def inverse_transform(self, X):
        check_is_fitted(self, "scale_")
        return np.asarray(X, dtype=np.float64) * self.scale_ + self.mean_
