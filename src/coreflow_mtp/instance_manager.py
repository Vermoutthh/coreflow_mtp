"""Capability-aware instance registry; no assumed legacy KV cache APIs."""

from dataclasses import dataclass, field
from typing import Any


CAPABILITY_KEYS = (
    "prefix_caching", "cache_state", "retained_gpu_entries", "cpu_swap",
    "cache_migration", "prefix_hashes",
    "predictive_cache",
    "cache_affinity_routing",
)


def normalize_capabilities(value: dict | None = None) -> dict[str, bool]:
    value = value or {}
    return {key: value.get(key) is True for key in CAPABILITY_KEYS}


@dataclass
class InstanceInfo:
    instance_id: str
    agent_id: str
    host: str
    port: int
    context_range: tuple[int, int] | None = None
    num_gpus: int = 1
    speculative_tokens: int = 0
    capabilities: dict[str, bool] = field(default_factory=normalize_capabilities)
    active_requests: int = 0
    local_inflight: int = 0
    status: str = "starting"
    model: str | None = None
    max_model_len: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def is_ready(self) -> bool:
        return self.status == "ready"


class InstanceManager:
    def __init__(self):
        self._instances: dict[str, InstanceInfo] = {}
        self._range_slots: dict[tuple[str, tuple[int, int]], int] = {}
        self._slot_ranges: dict[tuple[str, int], tuple[int, int]] = {}
        self._next_port = 8000

    def register(self, instance_id: str, agent_id: str, host: str,
                 port: int | None = None, context_range=None, num_gpus: int = 1,
                 speculative_tokens: int = 0, capabilities: dict | None = None,
                 model: str | None = None, max_model_len: int | None = None,
                 **metadata) -> InstanceInfo:
        if num_gpus < 1 or speculative_tokens < 0:
            raise ValueError("num_gpus must be positive and speculative_tokens nonnegative")
        if context_range is not None:
            context_range = tuple(map(int, context_range))
            if len(context_range) != 2 or not 0 <= context_range[0] < context_range[1]:
                raise ValueError("Expected a nonempty [lower, upper) context range")
            key = (agent_id, context_range)
            if key not in self._range_slots:
                slot = 1 + max((s for (a, s) in self._slot_ranges if a == agent_id), default=-1)
                self._range_slots[key] = slot
                self._slot_ranges[(agent_id, slot)] = context_range
        if port is None:
            port = self._next_port
            self._next_port += 1
        info = InstanceInfo(instance_id=instance_id, agent_id=agent_id, host=host,
                            port=int(port), context_range=context_range,
                            num_gpus=num_gpus, speculative_tokens=speculative_tokens,
                            capabilities=normalize_capabilities(capabilities),
                            model=model, max_model_len=max_model_len, metadata=metadata)
        self._instances[instance_id] = info
        return info

    def get(self, instance_id):
        return self._instances.get(instance_id)

    def set_ready(self, instance_id):
        if instance_id in self._instances:
            self._instances[instance_id].status = "ready"

    def mark_unavailable(self, instance_id):
        info = self.get(instance_id)
        if info:
            info.status = "unavailable"
        return info

    def remove(self, instance_id):
        self._instances.pop(instance_id, None)

    def all_instance_ids(self):
        return sorted(self._instances)

    def get_instances_for_agent(self, agent_id):
        return [self._instances[key] for key in sorted(self._instances)
                if self._instances[key].agent_id == agent_id and self._instances[key].is_ready]

    def get_instances_for_context(self, agent_id, slot):
        crange = self._slot_ranges.get((agent_id, slot))
        return [info for info in self.get_instances_for_agent(agent_id)
                if info.context_range == crange]

    def get_context_range_for_slot(self, agent_id, slot):
        return self._slot_ranges.get((agent_id, slot))

    def find_context_slot(self, agent_id, input_length):
        for (agent, slot), (lower, upper) in sorted(self._slot_ranges.items()):
            if agent == agent_id and lower <= input_length < upper:
                return slot
        return None

    def find_alternative(self, agent_id, slot, exclude_instance_id):
        # Never silently route a failed request to an incompatible context partition.
        return next((info for info in self.get_instances_for_context(agent_id, slot)
                     if info.instance_id != exclude_instance_id), None)
