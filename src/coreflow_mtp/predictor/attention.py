"""Attention latency predictor using linear regression on profile data."""

import json
from typing import Optional, Tuple

import numpy as np
from coreflow_mtp.predictor.linear import LinearRegression

from coreflow_mtp.predictor.base import BasePredictor


class AttentionPredictor(BasePredictor):
    """Predicts attention kernel latency (decode and prefill).

    Builds linear regression models from attn_profile.json.
    """

    def __init__(self, profile_path: str) -> None:
        super().__init__()
        self._profile_path = profile_path

    def _load_model(self) -> Tuple[LinearRegression, LinearRegression]:
        """Load and fit decode/prefill linear regression models."""
        with open(self._profile_path, "r") as f:
            data = json.load(f)

        decode_model = self._fit_model(data["decode"])
        prefill_model = self._fit_model(data["prefill"])
        return decode_model, prefill_model



    @staticmethod
    def _fit_model(raw_data: dict) -> LinearRegression:
        """Fit a linear model: latency ~ batch_size + total_tokens."""
        xs, ys = [], []
        for batch_size, seq_data in raw_data.items():
            for seq_len, latency in seq_data.items():
                xs.append([int(batch_size), int(seq_len)])
                ys.append(float(latency))

        model = LinearRegression()
        model.fit(np.array(xs), np.array(ys))
        return model

    @property
    def decode_model(self) -> LinearRegression:
        self._ensure_loaded()
        return self._model[0]

    @property
    def prefill_model(self) -> LinearRegression:
        self._ensure_loaded()
        return self._model[1]

    def predict(
        self,
        batch_size: int,
        total_tokens: int,
        mode: str = "decode",
    ) -> float:
        """Predict attention latency.

        Args:
            batch_size: Number of sequences in the batch.
            total_tokens: Total number of tokens (batch_size * avg_seq_len).
            mode: 'decode' or 'prefill'.

        Returns:
            Predicted attention latency in seconds.
        """
        self._ensure_loaded()
        model = self.decode_model if mode == "decode" else self.prefill_model
        return float(model.predict(np.array([[batch_size, total_tokens]]))[0])
