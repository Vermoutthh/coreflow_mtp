"""Explicit, reproducible configuration for one fixed-K serving instance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class InstanceConfig:
    model: str
    instance_id: str = "instance-0"
    agent_id: str = "default"
    host: str = "127.0.0.1"
    port: int = 8100
    context_lower: int = 0
    context_upper: int | None = None
    speculative_k: int = 0
    tensor_parallel_size: int = 1
    dtype: str = "bfloat16"
    max_model_len: int = 8192
    max_num_seqs: int = 16
    max_num_batched_tokens: int = 8192
    gpu_memory_utilization: float = 0.85
    kv_cache_memory_bytes: int | None = None
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = False
    enforce_eager: bool = True
    async_scheduling: bool = False
    language_model_only: bool = True
    seed: int = 17
    request_timeout_seconds: float = 600.0
    max_pending_requests: int = 128
    expected_vllm_version: str = "0.18.1"
    enable_cache_management: bool = False
    cpu_cache_bytes: int = 2147483648
    retained_gpu_fraction: float = 0.4
    cache_storage_root: str = "/root/nfs/sc_expansion/coreflow_mtp/run/cache_store"
    cache_policy_profile: str | None = None
    cache_affinity_routing: bool = False
    execution_trace: str | None = None

    def __post_init__(self) -> None:
        if not self.model.strip() or not self.instance_id.strip() or not self.agent_id.strip():
            raise ValueError("model, instance_id and agent_id must be nonempty")
        if self.speculative_k < 0:
            raise ValueError("speculative_k must be >= 0; K=0 disables speculative decoding")
        if self.context_lower < 0:
            raise ValueError("context_lower must be >= 0")
        if self.context_upper is not None and self.context_upper <= self.context_lower:
            raise ValueError("context range must be nonempty: [lower, upper)")
        if self.context_upper is not None and self.context_upper > self.max_model_len:
            raise ValueError("context_upper must not exceed max_model_len")
        if self.tensor_parallel_size < 1 or self.max_model_len < 2 or self.max_num_seqs < 1:
            raise ValueError("TP, maximum model length and maximum sequence count must be positive")
        if self.context_lower >= self.max_model_len:
            raise ValueError("context_lower must be below max_model_len")
        if self.max_num_batched_tokens < 1:
            raise ValueError("max_num_batched_tokens must be positive")
        if not self.enable_chunked_prefill and self.max_num_batched_tokens < self.max_model_len:
            raise ValueError("without chunked prefill the token budget must cover max_model_len")
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization must be between 0 and 1")
        if self.kv_cache_memory_bytes is not None and self.kv_cache_memory_bytes <= 0:
            raise ValueError("kv_cache_memory_bytes must be positive")
        if self.request_timeout_seconds <= 0 or self.max_pending_requests < 1:
            raise ValueError("request timeout and queue capacity must be positive")
        if not 1 <= self.port <= 65535:
            raise ValueError("invalid port")
        if self.dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ValueError("unsupported dtype")
        if self.enable_cache_management:
            if self.host not in {"127.0.0.1", "localhost"}:
                raise ValueError("unauthenticated cache control must bind to loopback")
            if not self.enable_prefix_caching:
                raise ValueError("explicit cache management requires native prefix caching")
            if self.async_scheduling:
                raise ValueError("cache transactions currently require synchronous V1 scheduling")
            if not self.language_model_only:
                raise ValueError("this cache transport is for text-only inputs")
            if self.cpu_cache_bytes < 1 or not 0 < self.retained_gpu_fraction < .9:
                raise ValueError("CPU-cache bytes and GPU residency fraction must be bounded")
        elif self.cache_policy_profile:
            raise ValueError("predictive cache policy requires cache management")
        if (self.cache_affinity_routing or self.execution_trace) and not self.enable_cache_management:
            raise ValueError("cache-affinity routing and engine trace require cache management")

    @property
    def context_range(self) -> list[int | None]:
        # A standalone instance without an explicit partition still advertises
        # a concrete range which the controller can safely register.
        return [self.context_lower, self.context_upper if self.context_upper is not None else self.max_model_len]

    def engine_kwargs(self) -> dict:
        """Pinned V1 configuration and explicit project extension paths.

        No dummy weights, forced outputs or replacement numerical kernels.
        """
        kwargs = {
            "model": self.model,
            "dtype": self.dtype,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enable_chunked_prefill": self.enable_chunked_prefill,
            "enable_prefix_caching": self.enable_prefix_caching,
            "enforce_eager": self.enforce_eager,
            "async_scheduling": self.async_scheduling,
            "language_model_only": self.language_model_only,
            "seed": self.seed,
            "load_format": "auto",
            "generation_config": "vllm",
            "trust_remote_code": False,
            "disable_log_stats": True,
        }
        if self.enforce_eager:
            kwargs["compilation_config"] = {"mode": 0}
        if self.kv_cache_memory_bytes is not None:
            kwargs["kv_cache_memory_bytes"] = self.kv_cache_memory_bytes
        if self.enable_prefix_caching and self.is_hybrid_model:
            # Qwen3.5's linear-attention and MTP state do not support "all".
            # align is still best-effort APC, not retained/pinned workflow KV.
            kwargs["mamba_cache_mode"] = "align"
        if self.speculative_k:
            kwargs["speculative_config"] = {
                "method": "mtp", "num_speculative_tokens": self.speculative_k,
            }
        if self.enable_cache_management:
            kwargs["scheduler_cls"] = "coreflow_mtp.cache.scheduler.CacheScheduler"
            kwargs["worker_extension_cls"] = "coreflow_mtp.cache.worker.CacheWorkerExtension"
            kwargs["additional_config"] = {"coreflow_cache": {
                "storage_root": str(Path(self.cache_storage_root).resolve()),
                "cpu_cache_bytes": self.cpu_cache_bytes,
                "gpu_fraction": self.retained_gpu_fraction,
                "compatibility_id": self.cache_compatibility_id,
                "execution_trace": self.execution_trace,
            }}
        return kwargs

    @property
    def is_hybrid_model(self) -> bool:
        path = Path(self.model) / "config.json"
        if not path.is_file():
            return False  # Do not force Mamba operations on an unknown dense model.
        raw = json.loads(path.read_text())
        text = raw.get("text_config", raw)
        return (any(kind in {"linear_attention", "mamba", "ssm"}
                    for kind in text.get("layer_types", []))
                or text.get("model_type") in {"mamba", "mamba2", "jamba", "zamba", "zamba2"})

    @property
    def cache_compatibility_id(self) -> str:
        model = Path(self.model)
        files = []
        if model.is_dir():
            for path in sorted(model.iterdir()):
                if path.name.endswith((".json", ".safetensors")):
                    stat = path.stat()
                    files.append((path.name, stat.st_size, stat.st_mtime_ns))
        return hashlib.sha256(json.dumps({"model": str(model.resolve()), "files": files,
            "dtype": self.dtype, "tp": self.tensor_parallel_size, "k": self.speculative_k,
            "vllm": self.expected_vllm_version}, sort_keys=True).encode()).hexdigest()

    def public_dict(self) -> dict:
        result = asdict(self)
        result["context_range"] = self.context_range
        result["speculative_tokens"] = self.speculative_k
        return result

    @property
    def backend_signature(self) -> str:
        payload = {"vllm_version": self.expected_vllm_version, "engine": self.engine_kwargs()}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
