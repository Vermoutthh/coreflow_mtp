"""Derive profiler geometry from a pinned, supported model checkpoint."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class ARModelGeometry:
    model: str
    architecture: str
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_size: int
    hidden_size: int
    intermediate_size: int
    max_model_len: int
    config_sha256: str
    block_size: int = 16
    dtype: str = "bfloat16"

    @classmethod
    def load(cls, model, dtype="bfloat16"):
        path = Path(model).resolve()
        content = (path / "config.json").read_bytes()
        raw = json.loads(content)
        if raw.get("architectures") != ["LlamaForCausalLM"]:
            raise ValueError("AR component model currently supports LlamaForCausalLM only; hybrid/MTP needs its own calibrated model")
        if dtype not in {"bfloat16", "float16"}:
            raise ValueError("AR profiling requires BF16 or FP16")
        heads, hidden = int(raw["num_attention_heads"]), int(raw["hidden_size"])
        return cls(str(path), "LlamaForCausalLM", int(raw["num_hidden_layers"]), heads,
            int(raw.get("num_key_value_heads", heads)), int(raw.get("head_dim", hidden // heads)),
            hidden, int(raw["intermediate_size"]), int(raw["max_position_embeddings"]),
            hashlib.sha256(content).hexdigest(), dtype=dtype)

    @property
    def kv_bytes_per_token(self):
        return self.num_layers * 2 * self.num_kv_heads * self.head_size * 2

    def public_dict(self):
        return {**asdict(self), "kv_bytes_per_token": self.kv_bytes_per_token}


def profiling_grid(smoke=False):
    if smoke:
        return {"batches": [1, 2, 4, 8], "contexts": [128, 512, 1024, 2048],
                "queries": [16, 32, 64, 128], "mixed_batches": [2, 4, 8],
                "long_contexts": [1024, 1536, 2048], "short_contexts": [128, 256, 512],
                "baseline_batches": [4, 8, 12, 16],
                "non_attention_batches": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                "swap_tokens": [128, 512, 1024]}
    return {"batches": list(range(1, 65)), "contexts": list(range(128, 10241, 128)),
            "queries": list(range(16, 513, 16)), "mixed_batches": list(range(2, 65)),
            "long_contexts": list(range(16000, 128001, 16000)),
            "short_contexts": list(range(1000, 8001, 1000)),
            "baseline_batches": list(range(4, 64, 4)),
            "non_attention_batches": [1, 2, 4, 8, 16] + list(range(32, 513, 32)),
            "swap_tokens": [128, 512, 1024, 2048]}
