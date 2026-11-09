"""Measured MTP capacity table -> original context DP and workflow greedy plan.

No capacity values are fabricated. Each row must already be normalized to
agent invocation/s using the workload demand proportions. Raw output token/s
must not be passed as capacity. The table is for fixed GPU/model/configuration;
replica capacities are assumed independent, as in the original CoreFlow DP.
"""

from dataclasses import dataclass
import json
import math

from coreflow_mtp.allocator.instance_orchestrator import InstanceOrchestrator
from coreflow_mtp.allocator.workflow_allocator import WorkflowAllocator


@dataclass(frozen=True)
class CapacityRecord:
    agent_id: str
    context_lower: int
    context_upper: int
    speculative_tokens: int
    normalized_invocations_per_second: float


class CapacityTable:
    def __init__(self, metadata, records, context_step=2000, max_context_length=200_000):
        required = {"model", "engine_version", "hardware", "dtype", "gpus_per_instance",
                    "workload_id", "provenance", "execution_config"}
        if not required <= set(metadata):
            raise ValueError(f"Missing capacity metadata: {sorted(required - set(metadata))}")
        if any(not str(metadata[key]).strip() for key in required):
            raise ValueError("Capacity metadata cannot be empty")
        if int(metadata["gpus_per_instance"]) < 1:
            raise ValueError("gpus_per_instance must be positive")
        execution = metadata["execution_config"]
        if not isinstance(execution, dict):
            raise ValueError("execution_config must describe the measured engine configuration")
        if any(not isinstance(execution.get(key), bool)
               for key in ("chunked_prefill", "prefix_caching")):
            raise ValueError("execution_config requires boolean chunked_prefill and prefix_caching")
        if not isinstance(execution.get("backend_signature"), str) or not execution["backend_signature"].strip():
            raise ValueError("execution_config requires a nonempty backend_signature")
        if context_step < 1 or max_context_length < 1 or max_context_length % context_step:
            raise ValueError("Invalid context grid")
        self.metadata = dict(metadata)
        self.context_step = int(context_step)
        self.max_context_length = int(max_context_length)
        self.records = []
        keys = set()
        for value in records:
            record = value if isinstance(value, CapacityRecord) else CapacityRecord(**value)
            if not isinstance(record.agent_id, str) or not record.agent_id.strip():
                raise ValueError("Capacity agent_id must be nonempty")
            if any(not isinstance(item, int) or isinstance(item, bool)
                   for item in (record.context_lower, record.context_upper, record.speculative_tokens)):
                raise ValueError("Capacity context bounds and K must be integers")
            if not (0 <= record.context_lower < record.context_upper <= max_context_length):
                raise ValueError("Invalid profiled context range")
            if record.context_lower % context_step or record.context_upper % context_step:
                raise ValueError("Capacity range is not aligned to the context grid")
            if (record.speculative_tokens < 0 or
                    not math.isfinite(record.normalized_invocations_per_second) or
                    record.normalized_invocations_per_second <= 0):
                raise ValueError("K and capacity must be valid positive finite observations")
            key = (record.agent_id, record.context_lower, record.context_upper, record.speculative_tokens)
            if key in keys:
                raise ValueError("Duplicate capacity row")
            keys.add(key)
            self.records.append(record)

    @classmethod
    def from_dict(cls, data):
        return cls(data["metadata"], data["records"], data.get("context_step", 2000),
                   data.get("max_context_length", 200_000))

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def best_for_agent(self, agent_id, allowed_k=None):
        best = {}
        for record in self.records:
            if record.agent_id != agent_id or (allowed_k is not None and record.speculative_tokens not in allowed_k):
                continue
            key = (record.context_lower // self.context_step, record.context_upper // self.context_step)
            old = best.get(key)
            if old is None or (record.normalized_invocations_per_second, -record.speculative_tokens) > (old.normalized_invocations_per_second, -old.speculative_tokens):
                best[key] = record
        return best


class ProfiledAgentAllocator:
    def __init__(self, table: CapacityTable, allowed_k=None):
        self.table = table
        self.allowed_k = allowed_k
        self.selected_records = {}
        self.orchestrator = InstanceOrchestrator(
            context_step=table.context_step, max_context_length=table.max_context_length,
            dp_gpu_step=int(table.metadata["gpus_per_instance"]))

    def allocate(self, total_num_gpus, agent_name, max_instance_group=3,
                 min_instance_group=1, **legacy_unused):
        records = self.table.best_for_agent(agent_name, self.allowed_k)
        self.selected_records[agent_name] = records
        result = self.orchestrator.solve(total_num_gpus,
            {key: row.normalized_invocations_per_second for key, row in records.items()},
            max_instance_group, min_instance_group)
        return result

    def selected_k(self, agent_id, context_lower, context_upper):
        records = self.selected_records.get(agent_id) or self.table.best_for_agent(agent_id, self.allowed_k)
        key = context_lower // self.table.context_step, context_upper // self.table.context_step
        if key not in records:
            raise ValueError("No profiled K for this context range")
        return records[key].speculative_tokens


def allocate_from_capacity_table(table: CapacityTable, total_num_gpus: int,
                                 agent_invocations: dict[str, float],
                                 max_instance_group=3, min_instance_group=1,
                                 allowed_k=None):
    """Return an allocation JSON suitable for the controller/instance launcher.

This preserves original CoreFlow DP/greedy heuristics, not an exact global
integer optimizer. Profiles outside the observed grid are never extrapolated.
    """
    if not agent_invocations or any(not math.isfinite(float(v)) or v <= 0 for v in agent_invocations.values()):
        raise ValueError("Positive finite invocation counts are required")
    if not 1 <= min_instance_group <= max_instance_group:
        raise ValueError("Invalid context partition count constraints")
    profiled = ProfiledAgentAllocator(table, allowed_k)

    class Invocations:
        def avg_num_invocations(self, agents):
            return {agent: agent_invocations[agent] for agent in agents}

    allocator = WorkflowAllocator(profiled, Invocations())
    gpus_per_instance = int(table.metadata["gpus_per_instance"])
    result = allocator.allocate(
        agents=sorted(agent_invocations), total_num_gpus=total_num_gpus,
        num_layers=0, instance_cached_tokens=0, dp_gpu_step=gpus_per_instance,
        max_instance_group=max_instance_group, min_instance_group=min_instance_group)
    if result.min_throughput <= 0:
        raise ValueError("Profile grid cannot cover the requested context domain for every agent")
    allocation = {}
    for agent, budget in result.gpu_allocation.items():
        assignment = result.dp_results[agent][budget]
        allocation[agent] = {"num_gpus": budget,
            "normalized_workflow_throughput": result.agent_throughput[agent],
            "instances": [{"context_lower": lower, "context_upper": upper,
                "num_instances": gpus // gpus_per_instance,
                "gpus_per_instance": gpus_per_instance,
                "speculative_tokens": profiled.selected_k(agent, lower, upper),
                "single_instance_normalized_capacity": throughput}
                for gpus, lower, upper, throughput in assignment.instances]}
    return {"schema_version": 1, "capacity_source": table.metadata.get("capacity_source", "measured_table"),
            "profile_metadata": table.metadata,
            "config": {"gpus_per_instance": gpus_per_instance, "model": table.metadata["model"]},
            "predicted_workflow_throughput": result.min_throughput,
            "agent_invocations": agent_invocations, "allocation": allocation}
