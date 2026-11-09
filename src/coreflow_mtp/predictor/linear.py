"""Small NumPy OLS model; no sklearn/scipy dependency in the engine environment."""
import numpy as np


class LinearRegression:
    def fit(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or not len(y):
            raise ValueError("OLS requires a nonempty feature matrix and one-dimensional targets")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("non-finite profiling observation")
        center = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale == 0] = 1
        design = np.column_stack([np.ones(len(x)), (x - center) / scale])
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        self.coef_ = coefficients[1:] / scale
        self.intercept_ = float(coefficients[0] - center @ self.coef_)
        self.rank_ = int(np.linalg.matrix_rank(design))
        return self

    def predict(self, x):
        return np.asarray(x, dtype=float) @ self.coef_ + self.intercept_
