"""Unified latency predictor combining attention, interference, and MLP models."""

import json
from typing import Dict, Optional

import numpy as np

from coreflow_mtp.predictor.attention import AttentionPredictor
from coreflow_mtp.predictor.interference import InterferencePredictor


class LatencyPredictor:
    """Unified predictor for total per-iteration latency.

    Composes attention, interference, and Llama non-attention layer components.
    All sub-models are lazy-loaded on first prediction.
    """

    def __init__(
        self,
        attn_profile_path: str,
        interference_profile_path: str,
        mlp_profile_path: str,
        *,
        allow_legacy_profiles: bool = False,
    ) -> None:
        if not allow_legacy_profiles:
            raise ValueError("Legacy Llama profiles are not an MTP capacity model. "
                             "Pass allow_legacy_profiles=True only for legacy reproduction.")
        self._attn = AttentionPredictor(attn_profile_path)
        self._interference = InterferencePredictor(interference_profile_path)
        self._mlp_profile_path = mlp_profile_path
        self._mlp_model: Optional[Dict[int, float]] = None
        self._mlp_loaded: bool = False

    def _load_mlp(self) -> Dict[int, float]:
        """Load per-layer non-attention latency table.

        The persisted file keeps the historical ``mlp_profile.json`` name, but
        each value is the latency of one Llama decoder layer excluding the
        attention kernel.
        """
        with open(self._mlp_profile_path, "r") as f:
            raw = json.load(f)
        return {int(bs): float(lat) for bs, lat in raw.items()}

    def _ensure_mlp_loaded(self) -> None:
        if not self._mlp_loaded:
            self._mlp_model = self._load_mlp()
            self._mlp_loaded = True

    def _predict_mlp(self, batch_size: int) -> float:
        self._ensure_mlp_loaded()
        if batch_size in self._mlp_model:
            return self._mlp_model[batch_size]
        keys = sorted(self._mlp_model)
        for key in keys:
            if key >= batch_size:
                return self._mlp_model[key]
        return self._mlp_model[keys[-1]]

    def predict_latency(
        self,
        num_layers: int,
        decode_batch_size: int,
        avg_decode_length: int,
        longest_decode_length: int = 0,
        prefill_query_len: int = 0,
        prefill_seq_len: int = 0,
        interference: bool = True,
    ) -> float:
        """Predict total per-iteration latency.

        Args:
            num_layers: Number of transformer layers.
            decode_batch_size: Number of sequences decoding in parallel.
            avg_decode_length: Average sequence length of decoding sequences.
            longest_decode_length: Length of the longest decoding sequence.
                Used for interference calculation.
            prefill_query_len: Number of new tokens to prefill (0 = decode-only).
            prefill_seq_len: KV cache length for the prefill sequence.
            interference: Whether to account for decode interference.

        Returns:
            Total iteration latency in seconds.
        """
        # Llama per-layer non-attention component.
        total_batch = decode_batch_size + prefill_query_len
        mlp_latency = num_layers * self._predict_mlp(total_batch)

        # Decode attention component
        decode_attn_latency = num_layers * self._attn.predict(
            batch_size=decode_batch_size,
            total_tokens=decode_batch_size * avg_decode_length,
            mode="decode",
        )

        # Prefill attention component (optional)
        if prefill_query_len != 0:
            prefill_attn_latency = num_layers * self._attn.predict(
                batch_size=prefill_query_len,
                total_tokens=prefill_seq_len,
                mode="prefill",
            )
        else:
            prefill_attn_latency = 0.0

        # Interference component
        if decode_batch_size == 1 or not interference:
            interference_latency = 0.0
        else:
            avg_remaining_length = (
                (avg_decode_length * decode_batch_size - longest_decode_length)
                / (decode_batch_size - 1)
            )
            interference_latency = num_layers * self._interference.predict(
                batch_size=decode_batch_size,
                large_seq_len=longest_decode_length,
                small_seq_len=int(avg_remaining_length),
            )

        return (
            mlp_latency
            + interference_latency
            + decode_attn_latency
            + prefill_attn_latency
        )
