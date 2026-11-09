"""Instance orchestrator for GPU-to-context-range assignment."""

from dataclasses import dataclass, field
import heapq
import math
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class MTPIntervalCapacity:
    """One explicitly profiled interval/K; raw rates are LLM requests/s.

    Traffic fractions MUST come from the complete target workload, not the
    balanced profiling samples. None means explicitly unsupported, not zero.
    """

    lower_bound: int
    upper_bound: int
    k: int
    requests_per_second: Optional[float]
    out_proportion: float
    query_proportion: float
    provenance: str
    unsupported_reason: str = ""

    def __post_init__(self):
        if (any(type(v) is not int for v in (self.lower_bound, self.upper_bound, self.k))
                or self.lower_bound < 0 or self.upper_bound <= self.lower_bound or self.k < 0):
            raise ValueError("valid initial context interval and nonnegative K required")
        if (not self.provenance or not math.isfinite(self.out_proportion)
                or not math.isfinite(self.query_proportion)
                or not 0 < self.out_proportion <= 1 or not 0 < self.query_proportion <= 1):
            raise ValueError("full-workload nonempty traffic fractions and provenance required")
        if self.requests_per_second is None:
            if not self.unsupported_reason:
                raise ValueError("unknown/unsupported capacity needs an explicit reason")
        elif (not math.isfinite(self.requests_per_second) or self.requests_per_second <= 0
              or self.unsupported_reason):
            raise ValueError("positive supported LLM capacity or explicit unsupported evidence required")

    @property
    def normalized_throughput(self):
        if self.requests_per_second is None:
            return None
        return self.requests_per_second * self.out_proportion / self.query_proportion


@dataclass
class InstanceAssignment:
    """Result of a single allocation candidate."""

    throughput: float = 0.0
    instances: List[Tuple[int, int, int, float]] = field(default_factory=list)
    """Each instance: (num_gpus, context_lower_bound, context_upper_bound, throughput)."""


class InstanceOrchestrator:
    """Determines GPU-to-context-range assignment.

    The algorithm follows the optimized-mapping path from the original
    allocator:

    1. Run a GPU-count-independent DP to select the best context partitioning
       for each ``(num_groups, context_upper)``.  The DP minimizes the sum of
       inverse single-instance throughput, which is equivalent to maximizing the
       balanced throughput of a continuously divisible GPU budget.
    2. For each concrete GPU budget, reconstruct each candidate partitioning
       and greedily assign instances to the currently slowest partition.
    """

    def __init__(
        self,
        context_step: int = 2000,
        max_context_length: int = 200_000,
        dp_gpu_step: int = 1,
    ) -> None:
        self._context_step = context_step
        self._max_context_length = max_context_length
        self._dp_gpu_step = dp_gpu_step

    @property
    def max_context_idx(self) -> int:
        return self._max_context_length // self._context_step

    def solve(
        self,
        total_num_gpus: int,
        instance_throughput_cache: Dict[Tuple[int, int], float],
        max_instance_group: int,
        min_instance_group: int = 1,
    ) -> Dict[int, InstanceAssignment]:
        """Return best assignments for every GPU count up to ``total_num_gpus``.

        Args:
            total_num_gpus: Maximum GPU budget.
            instance_throughput_cache: dict mapping (prev_idx, curr_idx) -> throughput.
            max_instance_group: Maximum number of instance groups.
            min_instance_group: Minimum number of instance groups for valid solutions.

        Returns:
            Dict mapping num_gpus -> InstanceAssignment with optimal throughput
            and list of (gpus, lower_bound, upper_bound, throughput).
        """
        dn = self._dp_gpu_step
        result: Dict[int, InstanceAssignment] = {}
        partition_plans = self._build_partition_plans(
            instance_throughput_cache=instance_throughput_cache,
            max_instance_group=max_instance_group,
        )

        for num_gpus in range(0, total_num_gpus + 1):
            best = InstanceAssignment()
            total_instances = num_gpus // dn
            for num_groups in range(min_instance_group, max_instance_group + 1):
                if total_instances < num_groups:
                    continue
                scheme = partition_plans.get(num_groups)
                if not scheme:
                    continue
                assignment = self._allocate_instances_greedily(
                    scheme=scheme,
                    total_num_gpus=num_gpus,
                    gpu_step=dn,
                )
                if assignment.throughput > best.throughput:
                    best = assignment
            result[num_gpus] = best

        return result

    def solve_mtp(
        self, total_num_gpus: int, interval_capacities, context_edges,
        max_instance_group: int = 3, min_instance_group: int = 1,
        k_values=(0,3,6), fixed_k=None,
    ) -> dict:
        """Thin K-selection/index adapter around ORIGINAL solve/DP/greedy.

        Require a complete table, including explicitly unsupported entries.
        Do not invent merged interval capacities or take balanced sample counts
        as routing proportions. This experimental entry never writes a launch
        allocation and does not upgrade conditional capacity to a guarantee.
        """
        edges, ks = tuple(context_edges), tuple(k_values)
        if (len(edges) < 2 or edges[0] != 0 or any(type(v) is not int for v in edges)
                or any(a >= b for a,b in zip(edges,edges[1:]))
                or not ks or len(set(ks)) != len(ks)
                or any(type(k) is not int or k < 0 for k in ks)
                or type(total_num_gpus) is not int or total_num_gpus < 0
                or type(self._dp_gpu_step) is not int or self._dp_gpu_step < 1
                or type(min_instance_group) is not int or type(max_instance_group) is not int
                or min_instance_group < 1 or max_instance_group < min_instance_group):
            raise ValueError("valid context edges, K candidates and GPU/group budgets required")
        if fixed_k is not None:
            if type(fixed_k) is not int or fixed_k not in ks:
                raise ValueError("fixed K must be one of the requested K candidates")
            ks = (fixed_k,)
        entries, traffic = {}, {}
        for row in interval_capacities:
            if not isinstance(row, MTPIntervalCapacity):
                raise ValueError("typed, traffic-normalized interval evidence required")
            key = (row.lower_bound, row.upper_bound, row.k)
            if key in entries:
                raise ValueError("duplicate interval/K capacity")
            if row.lower_bound not in edges or row.upper_bound not in edges:
                raise ValueError("capacity interval is not on the requested grid")
            interval = (row.lower_bound,row.upper_bound)
            fractions = (row.out_proportion,row.query_proportion)
            if interval in traffic and any(not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
                                           for a,b in zip(traffic[interval],fractions)):
                raise ValueError("different K values must use the SAME target workload traffic")
            traffic[interval] = fractions
            entries[key] = row
        required = [(a,b,k) for i,a in enumerate(edges[:-1]) for b in edges[i+1:] for k in ks]
        missing = [list(key) for key in required if key not in entries]
        result = dict(status="missing_capacity_profiles" if missing else "candidate",
            missing_interval_K=missing, context_edges=list(edges), k_values=list(ks), fixed_k=fixed_k,
            assignments=None, suitable_for_allocator=False,
            units="normalized_agent_invocations_per_second_NOT_workflow_throughput",
            algorithm="original_inverse_throughput_DP_then_original_greedy_instance_assignment",
            scope="conditional_interval_table_research_NOT_native_workflow_validation")
        if missing:
            return result
        best, cache = {}, {}
        positions = {value:i for i,value in enumerate(edges)}
        for a,b,_ in required:
            candidates = [entries[a,b,k] for k in ks if entries[a,b,k].requests_per_second is not None]
            if not candidates:
                continue
            row = max(candidates, key=lambda c:c.normalized_throughput)
            best[a,b] = row
            cache[positions[a],positions[b]] = row.normalized_throughput
        # Nonuniform real boundaries are merely indexed 0..m here. Neither the
        # original DP recurrence nor its integer-GPU greedy routine is changed.
        indexed = InstanceOrchestrator(context_step=1, max_context_length=len(edges)-1,
                                       dp_gpu_step=self._dp_gpu_step)
        assignments = indexed.solve(total_num_gpus, cache, max_instance_group, min_instance_group)
        output = {}
        for gpus, assignment in assignments.items():
            groups = []
            for count, lo_idx, hi_idx, normalized in assignment.instances:
                lo, hi = edges[lo_idx], edges[hi_idx]
                evidence = best[lo,hi]
                groups.append(dict(num_gpus=count, context_lower=lo, context_upper=hi,
                    K=evidence.k, LLM_requests_per_second=evidence.requests_per_second,
                    normalized_single_instance_throughput=normalized,
                    out_proportion=evidence.out_proportion, query_proportion=evidence.query_proportion,
                    provenance=evidence.provenance))
            output[gpus] = dict(status="candidate" if groups else "no_feasible_partition_or_GPU_budget",
                throughput=assignment.throughput if groups else None, instances=groups)
        result["assignments"] = output
        if not any(a["instances"] for a in output.values()):
            result["status"] = "no_feasible_partition_or_GPU_budget"
        return result

    def _build_partition_plans(
        self,
        instance_throughput_cache: Dict[Tuple[int, int], float],
        max_instance_group: int,
    ) -> Dict[int, List[Tuple[int, int, float]]]:
        """Run optimized-mapping DP and return plans ending at max context.

        The DP state ``dp[k][c]`` stores the minimum sum of inverse throughput
        needed to cover context range ``[0, c)`` with exactly ``k`` groups.  It
        is independent of the number of GPUs available later.
        """
        max_context_idx = self.max_context_idx
        valid_transitions: Dict[int, List[Tuple[int, float, float]]] = {}
        for (lo_idx, hi_idx), throughput in instance_throughput_cache.items():
            if throughput <= 0:
                continue
            valid_transitions.setdefault(hi_idx, []).append(
                (lo_idx, 1.0 / throughput, throughput))

        inf = float("inf")
        dp = [
            [inf] * (max_context_idx + 1)
            for _ in range(max_instance_group + 1)
        ]
        parent: Dict[Tuple[int, int], Tuple[int, float]] = {}
        dp[0][0] = 0.0

        for num_groups in range(1, max_instance_group + 1):
            for context_idx in range(1, max_context_idx + 1):
                for prev_idx, cost, throughput in valid_transitions.get(
                        context_idx, []):
                    prev_cost = dp[num_groups - 1][prev_idx]
                    if prev_cost == inf:
                        continue
                    new_cost = prev_cost + cost
                    if new_cost < dp[num_groups][context_idx]:
                        dp[num_groups][context_idx] = new_cost
                        parent[(num_groups, context_idx)] = (
                            prev_idx, throughput)

        plans: Dict[int, List[Tuple[int, int, float]]] = {}
        for num_groups in range(1, max_instance_group + 1):
            if dp[num_groups][max_context_idx] == inf:
                continue
            scheme = self._reconstruct_scheme(
                parent=parent,
                num_groups=num_groups,
                context_idx=max_context_idx,
            )
            if scheme:
                plans[num_groups] = scheme
        return plans

    @staticmethod
    def _reconstruct_scheme(
        parent: Dict[Tuple[int, int], Tuple[int, float]],
        num_groups: int,
        context_idx: int,
    ) -> List[Tuple[int, int, float]]:
        scheme: List[Tuple[int, int, float]] = []
        cur_groups = num_groups
        cur_context = context_idx
        while cur_groups > 0:
            entry = parent.get((cur_groups, cur_context))
            if entry is None:
                return []
            prev_context, throughput = entry
            scheme.append((prev_context, cur_context, throughput))
            cur_context = prev_context
            cur_groups -= 1
        if cur_context != 0:
            return []
        scheme.reverse()
        return scheme

    def _allocate_instances_greedily(
        self,
        scheme: List[Tuple[int, int, float]],
        total_num_gpus: int,
        gpu_step: int,
    ) -> InstanceAssignment:
        """Assign instances by repeatedly improving the bottleneck segment."""
        num_partitions = len(scheme)
        total_instances = total_num_gpus // gpu_step
        if total_instances < num_partitions:
            return InstanceAssignment()

        counts = [1] * num_partitions
        heap = [(scheme[idx][2], idx) for idx in range(num_partitions)]
        heapq.heapify(heap)

        for _ in range(total_instances - num_partitions):
            _cur_tp, idx = heapq.heappop(heap)
            counts[idx] += 1
            heapq.heappush(heap, (counts[idx] * scheme[idx][2], idx))

        throughput = min(
            counts[idx] * scheme[idx][2] for idx in range(num_partitions))
        instances = [
            (
                counts[idx] * gpu_step,
                lo_idx * self._context_step,
                hi_idx * self._context_step,
                single_tp,
            )
            for idx, (lo_idx, hi_idx, single_tp) in enumerate(scheme)
        ]
        return InstanceAssignment(throughput=throughput, instances=instances)
