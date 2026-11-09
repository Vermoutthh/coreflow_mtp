"""Engine-independent allocation and execution metadata.

No Llama architecture, KV-token capacity or profiling numbers are defaults for
Qwen3.5. Supply measured capacity data for the actual execution configuration.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AllocatorConfig:
    context_step: int = 2000
    max_context_length: int = 200_000
    gpus_per_instance: int = 1
    max_instance_group: int = 3
    min_instance_group: int = 1

    def __post_init__(self):
        if self.context_step <= 0 or self.max_context_length <= 0:
            raise ValueError("Context ranges must be positive")
        if self.max_context_length % self.context_step:
            raise ValueError("max_context_length must be divisible by context_step")
        if self.gpus_per_instance <= 0:
            raise ValueError("gpus_per_instance must be positive")


@dataclass(frozen=True)
class ProfileMetadata:
    model: str
    engine_version: str
    hardware: str
    dtype: str
    gpus_per_instance: int
    chunked_prefill: bool
    provenance: str


PROJECT_ROOT = Path(__file__).resolve().parents[2]
