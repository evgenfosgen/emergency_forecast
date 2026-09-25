"""сигмоидальная калибровка модельной вероятности"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


class SigmoidProbabilityCalibrator(BaseEstimator, ClassifierMixin):
    """калибрует вероятность логистической регрессией на logit(p)."""

    def __init__(self, *, c: float = 1_000_000.0, random_state: int = 0) -> None:
        self.c = float(c)
        self.random_state = int(random_state)

    @staticmethod
    def _logit(probability: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(
            probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
        return np.log(clipped / (1.0 - clipped))

    def fit(self, raw_probability: Any, y: Any) -> "SigmoidProbabilityCalibrator":
        x = self._logit(np.asarray(raw_probability,
                        dtype=np.float64)).reshape(-1, 1)
        target = np.asarray(y, dtype=np.int8)
        if x.shape[0] != target.shape[0]:
            raise ValueError("raw_probability и y имеют разную длину")
        if np.unique(target).size < 2:
            raise ValueError("для сигмоидальной калибровки нужны оба класса")
        self.model_ = LogisticRegression(
            C=self.c,
            fit_intercept=True,
            max_iter=1000,
            random_state=self.random_state,
            solver="lbfgs",
        )
        self.model_.fit(x, target)
        self.classes_ = np.array([0, 1], dtype=np.int8)
        self.n_features_in_ = 1
        return self

    def predict_proba(self, raw_probability: Any) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("калибратор еще не обучен")
        x = self._logit(np.asarray(raw_probability,
                        dtype=np.float64)).reshape(-1, 1)
        return self.model_.predict_proba(x)

    def transform(self, raw_probability: Any) -> np.ndarray:
        return np.asarray(self.predict_proba(raw_probability)[:, 1], dtype=np.float64)
