"""Checked K=0 profile bundle -> original hierarchical latency model.

Unlike allow_legacy_profiles, this path verifies new-runtime provenance and
model geometry. It deliberately rejects hybrid models and speculative K.
"""
import hashlib
import json
from pathlib import Path

from coreflow_mtp.predictor.latency import LatencyPredictor
from coreflow_mtp.profiler.config import ARModelGeometry


class ARLatencyPredictor(LatencyPredictor):
    def __init__(self, bundle_path, *, speculative_k=0):
        path = Path(bundle_path).resolve()
        self.metadata = json.loads(path.read_text())
        meta = self.metadata
        if (speculative_k != 0 or meta.get("speculative_k") != 0
                or meta.get("kind") != "coreflow_ar_components"
                or meta.get("engine_version") != "0.18.1"):
            raise ValueError("this checked predictor accepts only vLLM 0.18.1 AR/K=0 component profiles")
        if meta.get("tensor_parallel_size") != 1 or meta.get("attention_backend") != "FLASH_ATTN_FA2":
            raise ValueError("uncalibrated TP/attention backend")
        geometry = ARModelGeometry.load(meta["geometry"]["model"], meta["dtype"])
        if geometry.public_dict() != meta["geometry"]:
            raise ValueError("model configuration changed after profiling")
        self.geometry = geometry
        self.profile_paths = {}
        for name, record in meta["files"].items():
            source = (path.parent / record["path"]).resolve()
            if not source.is_relative_to(path.parent):
                raise ValueError("profile component must be inside its bundle directory")
            if hashlib.sha256(source.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("profile component checksum mismatch")
            self.profile_paths[name] = source
        super().__init__(str(self.profile_paths["attn_profile.json"]),
                         str(self.profile_paths["interference_profile.json"]),
                         str(self.profile_paths["mlp_profile.json"]), allow_legacy_profiles=True)
        self.swap_profile = json.loads(self.profile_paths["swap_profile.json"].read_text())

    @property
    def max_decode_batch(self):
        return self.metadata["fit_domain"]["max_decode_batch"]

    @property
    def supported_decode_batches(self):
        return self.metadata.get("profiled_decode_batches") or sorted(
            int(value) for value in json.loads(self.profile_paths["attn_profile.json"].read_text())["decode"])

    def swap_latency(self, cached_tokens, direction="h2d"):
        if cached_tokens <= 0:
            return 0.
        fit = self.swap_profile[direction]
        return fit["intercept_seconds"] + fit["seconds_per_byte"] * cached_tokens * self.geometry.kv_bytes_per_token

    def predict_latency(self, num_layers, decode_batch_size, avg_decode_length,
                        longest_decode_length=0, prefill_query_len=0, prefill_seq_len=0,
                        interference=True):
        domain = self.metadata["fit_domain"]
        if (num_layers != self.geometry.num_layers or decode_batch_size not in self.supported_decode_batches
                or prefill_query_len > domain["max_query"]
                or decode_batch_size + prefill_query_len > domain["max_non_attention_tokens"]
                or max(avg_decode_length, longest_decode_length, prefill_seq_len) > domain["max_context"]):
            raise ValueError("request lies outside calibrated AR component domain")
        return max(0., super().predict_latency(num_layers, decode_batch_size,
            avg_decode_length, longest_decode_length, prefill_query_len, prefill_seq_len, interference))
