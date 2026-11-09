"""Workflow-level allocator: greedily assigns instances across agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from coreflow_mtp.allocator.instance_orchestrator import InstanceAssignment
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coreflow_mtp.allocator.agent_allocator import AgentAllocator
    from coreflow_mtp.allocator.request_analyzer import RequestAnalyzer


@dataclass
class WorkflowAllocation:
    """Complete allocation result for a multi-agent workflow."""

    # Per-agent GPU allocation
    gpu_allocation: Dict[str, int] = field(default_factory=dict)

    # Per-agent normalized throughput (requests/sec adjusted for invocations)
    agent_throughput: Dict[str, float] = field(default_factory=dict)

    # Per-agent allocation results (GPU -> InstanceAssignment)
    dp_results: Dict[str, Dict[int, InstanceAssignment]] = field(default_factory=dict)

    # Min throughput across agents (the bottleneck)
    min_throughput: float = 0.0

    # Agent invocations per workflow trace
    agent_invocations: Dict[str, float] = field(default_factory=dict)

    # Number of iterations taken to converge
    iterations: int = 0


class WorkflowAllocator:
    """Top-level allocator for multi-agent workflow GPU distribution.

    Starts with one instance per agent.  It then repeatedly gives one more
    instance to the agent with the lowest normalized throughput until the GPU
    budget is exhausted.
    """

    def __init__(
        self,
        agent_allocator: AgentAllocator,
        request_analyzer: RequestAnalyzer,
        max_iterations: int = 100,
    ) -> None:
        self._agent_allocator = agent_allocator
        self._analyzer = request_analyzer
        self._max_iterations = max_iterations

    def allocate(
        self,
        agents: List[str],
        total_num_gpus: int,
        num_layers: int,
        instance_cached_tokens: int,
        dp_gpu_step: int = 1,
        max_instance_group: int = 3,
        min_instance_group: int = 1,
        interference: bool = False,
        reuse: bool = False,
        initial_gpus_per_agent: Optional[List[int]] = None,
        verbose: bool = False,
    ) -> WorkflowAllocation:
        """Run the greedy per-instance workflow allocation algorithm.

        Args:
            agents: Ordered list of agent names.
            total_num_gpus: Total GPU budget.
            num_layers: Transformer layers per model.
            instance_cached_tokens: KV cache capacity per instance.
            dp_gpu_step: GPU granularity for allocation (gpus_per_instance).
                All GPU adjustments are made in multiples of this value.
            max_instance_group: Max instance groups per agent.
            min_instance_group: Min instance groups per agent.
            interference: Account for interference.
            reuse: Enable KV cache reuse.
            initial_gpus_per_agent: Legacy argument; ignored by the current
                greedy strategy.
            verbose: Print per-iteration progress.

        Returns:
            WorkflowAllocation with final GPU distribution and throughputs.
        """
        G = dp_gpu_step  # shorthand for readability
        if not agents:
            return WorkflowAllocation()
        if G <= 0:
            raise ValueError("dp_gpu_step must be positive")
        if total_num_gpus % G != 0:
            raise ValueError(
                "total_num_gpus must be divisible by dp_gpu_step/gpus_per_instance"
            )
        if total_num_gpus < len(agents) * G:
            raise ValueError(
                "total_num_gpus is too small to allocate one instance per agent"
            )

        # Compute agent invocation counts
        agent_invocations = self._analyzer.avg_num_invocations(agents)

        # Build the per-agent throughput/allocation curve once.
        dp_results: Dict[str, Dict[int, InstanceAssignment]] = {}
        for agent in agents:
            dp_results[agent] = self._agent_allocator.allocate(
                num_layers=num_layers,
                total_num_gpus=total_num_gpus,
                instance_cached_tokens=instance_cached_tokens,
                agent_name=agent,
                max_instance_group=max_instance_group,
                min_instance_group=min_instance_group,
                interference=interference,
                reuse=reuse,
            )

        # Start with one instance per agent, then allocate one instance at a
        # time to the current workflow bottleneck.
        gpus = [G] * len(agents)
        allocated_gpus = len(agents) * G
        iterations = 0

        while allocated_gpus < total_num_gpus:
            throughputs = self._normalized_throughputs(
                agents=agents,
                gpus=gpus,
                dp_results=dp_results,
                agent_invocations=agent_invocations,
            )
            if verbose:
                print(
                    f"Iter {iterations}: gpus={gpus}, "
                    f"throughputs={[f'{t:.2f}' for t in throughputs]}"
                )

            bottleneck_idx = min(
                range(len(agents)),
                key=lambda idx: (throughputs[idx], agents[idx]),
            )
            gpus[bottleneck_idx] += G
            allocated_gpus += G
            iterations += 1

        # Build final result
        final_throughputs = dict(zip(
            agents,
            self._normalized_throughputs(
                agents=agents,
                gpus=gpus,
                dp_results=dp_results,
                agent_invocations=agent_invocations,
            ),
        ))

        return WorkflowAllocation(
            gpu_allocation=dict(zip(agents, gpus)),
            agent_throughput=final_throughputs,
            dp_results=dp_results,
            min_throughput=min(final_throughputs.values()),
            agent_invocations=agent_invocations,
            iterations=iterations,
        )

    @staticmethod
    def _normalized_throughputs(
        agents: List[str],
        gpus: List[int],
        dp_results: Dict[str, Dict[int, InstanceAssignment]],
        agent_invocations: Dict[str, float],
    ) -> List[float]:
        throughputs: List[float] = []
        for i, agent in enumerate(agents):
            invocations = float(agent_invocations.get(agent, 0.0))
            if invocations <= 0:
                throughputs.append(0.0)
                continue
            throughputs.append(dp_results[agent][gpus[i]].throughput /
                               invocations)
        return throughputs
