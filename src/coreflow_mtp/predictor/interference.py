"""Interference latency predictor for batched decode."""

import json
from typing import Dict, Optional

import numpy as np
from coreflow_mtp.predictor.linear import LinearRegression

from coreflow_mtp.predictor.base import BasePredictor


class InterferencePredictor(BasePredictor):
    """Predicts decode interference latency between long and short sequences.

    Builds per-batch-size linear models from interference_profile.json.
    """

    def __init__(self, profile_path: str) -> None:
        super().__init__()
        self._profile_path = profile_path

    def _load_model(self) -> Dict[int, LinearRegression]:
        """Load and fit per-batch-size linear models."""
        with open(self._profile_path, "r") as f:
            data = json.load(f)

        models: Dict[int, LinearRegression] = {}
        for batch_size_str, batch_data in data.items():
            batch_size = int(batch_size_str)
            xs, ys = [], []
            for large_seq_len, seq_data in batch_data.items():
                for small_seq_len, latency in seq_data.items():
                    xs.append([int(large_seq_len), int(small_seq_len)])
                    ys.append(float(latency))

            model = LinearRegression()
            model.fit(np.array(xs), np.array(ys))
            models[batch_size] = model

        return models

    def predict(
        self,
        batch_size: int,
        large_seq_len: int,
        small_seq_len: int,
    ) -> float:
        """Predict interference latency.

        Args:
            batch_size: Total batch size (including the long sequence).
            large_seq_len: Length of the longest sequence in the batch.
            small_seq_len: Average remaining length of other sequences.

        Returns:
            Predicted interference latency in seconds. Non-negative.
        """
        self._ensure_loaded()
        # Cap batch_size key to 50 for sparse profiles
        key = batch_size if batch_size in self._model else (50 if batch_size > 50 else batch_size)
        if key not in self._model:
            return 0.0
        return max(
            0.0,
            float(
                self._model[key].predict(
                    np.array([[large_seq_len, small_seq_len]])
                )[0]
            ),
        )

    def get_model(self, batch_size: int) -> Optional[LinearRegression]:
        """Get the linear model for a specific batch size, if available."""
        self._ensure_loaded()
        key = batch_size if batch_size in self._model else (50 if batch_size > 50 else batch_size)
        return self._model.get(key)
