"""Candidate guard around the unchanged original context-partition solver.

Retaining fixed-K candidates guarantees a lower bound, NOT integer optimality.
It changes neither the original DP nor the budget/capacity prediction.
"""
from copy import deepcopy

from .instance_orchestrator import InstanceOrchestrator


def select_mtp_partitions(total_num_gpus, interval_capacities, context_edges,
                          *, k_values=(0,3,6), max_instance_group=3,
                          min_instance_group=1, orchestrator=None):
    orchestrator=orchestrator or InstanceOrchestrator()
    entries=list(interval_capacities)
    k_values,context_edges=tuple(k_values),tuple(context_edges)
    common=dict(max_instance_group=max_instance_group,min_instance_group=min_instance_group,
                k_values=k_values)
    candidates={"adaptive_original":orchestrator.solve_mtp(
        total_num_gpus,entries,context_edges,**common)}
    if candidates["adaptive_original"]["status"]=="missing_capacity_profiles":
        return dict(status="missing_capacity_profiles",assignments=None,candidates=candidates,
            suitable_for_allocator=False,integer_optimality_claimed=False)
    for k in k_values:
        candidates[f"fixed_K{k}"]=orchestrator.solve_mtp(
            total_num_gpus,entries,context_edges,fixed_k=k,**common)
    assignments={}
    for gpu in range(total_num_gpus+1):
        available=[(label,result["assignments"][gpu]) for label,result in candidates.items()
            if result.get("assignments") is not None
            and result["assignments"][gpu]["throughput"] is not None]
        if available:
            label,best=max(available,key=lambda item:item[1]["throughput"])
            assignments[gpu]=dict(deepcopy(best),selected_candidate=label)
        else:
            assignments[gpu]=dict(status="no_feasible_partition_or_GPU_budget",throughput=None,
                instances=[],selected_candidate=None)
    return dict(status="candidate" if any(a["instances"] for a in assignments.values())
                else "no_feasible_partition_or_GPU_budget",
        assignments=assignments,candidates=candidates,suitable_for_allocator=False,
        units="normalized_agent_invocations_per_second_NOT_workflow_throughput",
        algorithm="unchanged_original_DP_and_greedy_retain_all_fixed_K_candidates",
        integer_optimality_claimed=False)
