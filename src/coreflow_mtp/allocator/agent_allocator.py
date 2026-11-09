"""Agent-level allocation; estimator must match the real execution backend."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from coreflow_mtp.allocator.instance_orchestrator import (
    InstanceAssignment,
    InstanceOrchestrator,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coreflow_mtp.allocator.throughput_estimator import ThroughputEstimator


class AgentAllocator:
    """Allocates GPU resources for a single agent.

    Runs throughput estimation across context ranges, then solves
    the instance group assignment.
    """

    def __init__(
        self,
        throughput_estimator: ThroughputEstimator,
        instance_orchestrator: InstanceOrchestrator,
    ) -> None:
        self._estimator = throughput_estimator
        self._orchestrator = instance_orchestrator

    def allocate(
        self,
        num_layers: int,
        total_num_gpus: int,
        instance_cached_tokens: int,
        agent_name: str,
        max_instance_group: int = 3,
        min_instance_group: int = 1,
        interference: bool = True,
        reuse: bool = True,
    ) -> Dict[int, InstanceAssignment]:
        """Compute throughput-vs-GPU curve for a single agent.

        Args:
            num_layers: Number of transformer layers.
            total_num_gpus: Maximum GPU budget to consider.
            instance_cached_tokens: KV cache capacity per instance.
            agent_name: Name of the agent.
            max_instance_group: Maximum instance groups.
            min_instance_group: Minimum instance groups.
            interference: Account for decode interference.
            reuse: Enable KV cache reuse.

        Returns:
            Dict mapping num_gpus -> InstanceAssignment.
        """
        # 1. Estimate throughput for all context range partitions
        instance_tp_cache = self._estimator.attain_instance_throughput(
            num_layers=num_layers,
            instance_cached_tokens=instance_cached_tokens,
            agent_name=agent_name,
            interference=interference,
            reuse=reuse,
        )

        # 2. Solve the instance group assignment
        dp_result = self._orchestrator.solve(
            total_num_gpus=total_num_gpus,
            instance_throughput_cache=instance_tp_cache,
            max_instance_group=max_instance_group,
            min_instance_group=min_instance_group,
        )

        return dp_result
