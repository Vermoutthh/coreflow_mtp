"""Throughput estimator for vLLM instances using binary search."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import cached_property
import math
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from coreflow_mtp.predictor.latency import LatencyPredictor
    from coreflow_mtp.allocator.request_analyzer import RequestAnalyzer


@dataclass(frozen=True)
class MTPRequestStats:
    """Per-request work; output includes the initial prefill-generated token."""

    mean_prefill_tokens: float
    mean_output_tokens: float
    mean_initial_output_tokens: float = 1.0
    serial_cache_seconds_per_request: float = 0.0

    @property
    def remaining_decode_tokens(self) -> float:
        return self.mean_output_tokens - self.mean_initial_output_tokens

    def validate(self):
        values = (self.mean_prefill_tokens, self.mean_output_tokens,
                  self.mean_initial_output_tokens, self.serial_cache_seconds_per_request)
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("MTP request statistics must be finite and nonnegative")
        if self.remaining_decode_tokens < 0:
            raise ValueError("initial output cannot exceed total output")
        if self.mean_prefill_tokens == 0 and self.mean_initial_output_tokens:
            raise ValueError("without prefill, do not count an initial prefill output")
        if not (self.mean_prefill_tokens or self.remaining_decode_tokens or self.serial_cache_seconds_per_request):
            raise ValueError("zero-work requests have no meaningful service capacity")


@dataclass(frozen=True)
class MTPResourceLimits:
    """Each key names ONE physical page pool, not an aliased logical group."""

    max_num_batched_tokens: int
    max_num_seqs: int
    available_pages: Dict[str, int]
    non_kv_headroom_bytes: int

    def validate(self):
        for value in (self.max_num_batched_tokens, self.max_num_seqs):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("positive integer engine limits required")
        if not self.available_pages or any(not isinstance(key, str) or not key or not isinstance(n, int) or isinstance(n, bool) or n < 0
                                           for key, n in self.available_pages.items()):
            raise ValueError("explicit physical-pool page budgets required")
        if (not isinstance(self.non_kv_headroom_bytes, int) or isinstance(self.non_kv_headroom_bytes, bool)
                or self.non_kv_headroom_bytes < 0):
            raise ValueError("non-KV workspace headroom must be nonnegative bytes")


@dataclass(frozen=True)
class MTPExecutionMode:
    """One FIXED pair of pure-decode/mixed shapes throughout one bisection.

    U values are total committed DECODE tokens per batch-round. In particular,
    U_mixed excludes the initial token of the newly prefilling request. Page
    demands include live hybrid/MTP states; workspace is additional to weights
    and preallocated KV pools. Missing resource evidence is not zero demand.
    """

    name: str
    k: int
    decode_batch_size: int
    decode_seconds: float
    decode_tokens_per_round: float
    mixed_decode_batch_size: int = 0
    mixed_seconds: Optional[float] = None
    mixed_decode_tokens_per_round: float = 0.0
    scheduled_prefill_tokens: int = 0
    effective_prefill_tokens_per_round: float = 0.0
    prefill_request_count: int = 1
    decode_required_pages: Dict[str, int] = field(default_factory=dict)
    mixed_required_pages: Dict[str, int] = field(default_factory=dict)
    decode_workspace_bytes: Optional[int] = None
    mixed_workspace_bytes: Optional[int] = None
    profiling_supported: bool = False
    native_shape_supported: bool = False
    preemption_free: bool = False
    provenance: str = ""

    def unsupported_reasons(self, stats, limits):
        reasons = []
        for name in ("profiling_supported", "native_shape_supported", "preemption_free"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be an explicit boolean")
        if not isinstance(self.provenance, str):
            raise ValueError("provenance must be a string")
        for name in ("k", "decode_batch_size", "mixed_decode_batch_size", "scheduled_prefill_tokens", "prefill_request_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not self.name or self.decode_batch_size < 1:
            raise ValueError("a named positive decode batch is required")
        if not math.isfinite(self.decode_seconds) or self.decode_seconds <= 0:
            raise ValueError("positive finite decode seconds required")
        if not 0 <= self.decode_tokens_per_round <= self.decode_batch_size * (self.k + 1):
            raise ValueError("decode yield must be a TOTAL batch yield within B*(K+1)")
        if not math.isfinite(self.decode_tokens_per_round):
            raise ValueError("nonfinite decode yield")
        if not self.profiling_supported:
            reasons.append("missing_cost_or_yield_profile")
        if not self.native_shape_supported:
            reasons.append("native_shape_not_verified")
        if not self.preemption_free:
            reasons.append("preemption_or_recompute_not_covered")
        if not self.provenance.strip():
            reasons.append("missing_profile_provenance")
        if self.decode_batch_size > limits.max_num_seqs or self.decode_batch_size * (self.k + 1) > limits.max_num_batched_tokens:
            reasons.append("decode_shape_exceeds_engine_limits")
        stages = [("decode", self.decode_required_pages, self.decode_workspace_bytes)]
        if stats.mean_prefill_tokens > 0:
            if self.mixed_seconds is None or not math.isfinite(self.mixed_seconds) or self.mixed_seconds <= 0:
                reasons.append("missing_mixed_latency")
            if (not math.isfinite(self.effective_prefill_tokens_per_round)
                    or self.effective_prefill_tokens_per_round <= 0
                    or self.effective_prefill_tokens_per_round > self.scheduled_prefill_tokens):
                reasons.append("invalid_effective_prefill_progress")
            if (not math.isfinite(self.mixed_decode_tokens_per_round)
                    or not 0 <= self.mixed_decode_tokens_per_round <= self.mixed_decode_batch_size * (self.k + 1)):
                raise ValueError("mixed yield must exclude prefill output and stay within B_mixed*(K+1)")
            if (self.prefill_request_count < 1 or self.mixed_decode_batch_size + self.prefill_request_count > limits.max_num_seqs
                    or self.scheduled_prefill_tokens + self.mixed_decode_batch_size * (self.k + 1) > limits.max_num_batched_tokens):
                reasons.append("mixed_shape_exceeds_engine_limits")
            stages.append(("mixed", self.mixed_required_pages, self.mixed_workspace_bytes))
        for stage, required, workspace in stages:
            if set(required) != set(limits.available_pages):
                reasons.append(f"{stage}_physical_pool_profile_missing_or_mismatched")
            else:
                if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in required.values()):
                    raise ValueError("page requirements must be nonnegative integer physical counts")
                if any(required[pool] > total for pool, total in limits.available_pages.items()):
                    reasons.append(f"{stage}_page_budget_exceeded")
            if workspace is None:
                reasons.append(f"{stage}_workspace_profile_missing")
            elif not isinstance(workspace, int) or isinstance(workspace, bool) or workspace < 0:
                raise ValueError("workspace requirements must be nonnegative integer bytes")
            elif workspace > limits.non_kv_headroom_bytes:
                reasons.append(f"{stage}_workspace_budget_exceeded")
        return reasons


@dataclass(frozen=True)
class MTPRoundState:
    """One witnessed native round. Query widths are not committed output.

    States are weighted within decode-only and prefill-bearing populations
    separately. A prefill-bearing round may have zero decoding requests.
    Initial prefill output is recorded but is never decode yield.
    """

    stage: str
    weight: float
    seconds: float
    decode_contexts: tuple
    decode_query_lens: tuple
    prefill_query_lens: tuple
    decode_tokens: float
    effective_prefill_tokens: float
    initial_output_tokens: float
    required_pages: Dict[str, int]
    preemptions: int = 0
    recompute: bool = False


@dataclass(frozen=True)
class MTPExecutionDistribution:
    """Fixed EMPIRICAL state distribution, not a fake fixed mean batch.

    Moments give E[U]/E[T], not E[U/T]. Each state must meet native query,
    sequence and physical-pool constraints on its own. Feasible observed
    states do not prove FCFS can preserve their mixture at a new arrival rate.
    Workspace is a conservative measured bound for the entire calibration.
    """

    name: str
    k: int
    states: tuple
    workspace_bytes: Optional[int]
    profiling_supported: bool
    native_shape_supported: bool
    preemption_free: bool
    provenance: str

    @cached_property
    def moments(self):
        result = {}
        for stage in ("decode", "prefill"):
            states = [r for r in self.states if r.stage == stage]
            total = sum(r.weight for r in states)
            result[stage] = {"weight": total}
            for field_name in ("seconds", "decode_tokens", "effective_prefill_tokens", "initial_output_tokens"):
                result[stage][field_name] = (
                    sum(r.weight * getattr(r, field_name) for r in states) / total if total else None)
        return result

    @property
    def decode_seconds(self):
        return self.moments["decode"]["seconds"]

    @property
    def decode_tokens_per_round(self):
        return self.moments["decode"]["decode_tokens"]

    @property
    def mixed_seconds(self):
        return self.moments["prefill"]["seconds"]

    @property
    def mixed_decode_tokens_per_round(self):
        return self.moments["prefill"]["decode_tokens"] or 0.

    @property
    def effective_prefill_tokens_per_round(self):
        return self.moments["prefill"]["effective_prefill_tokens"] or 0.

    def unsupported_reasons(self, stats, limits):
        reasons = []
        if not self.name or not isinstance(self.k, int) or isinstance(self.k, bool) or self.k < 0:
            raise ValueError("named distribution and nonnegative integer K required")
        for field_name in ("profiling_supported", "native_shape_supported", "preemption_free"):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be an explicit boolean")
        if not isinstance(self.provenance, str):
            raise ValueError("provenance must be a string")
        if not self.profiling_supported:
            reasons.append("missing_cost_or_yield_profile")
        if not self.native_shape_supported:
            reasons.append("native_shape_not_verified")
        if not self.preemption_free:
            reasons.append("preemption_or_recompute_not_covered")
        if not self.provenance.strip():
            reasons.append("missing_profile_provenance")
        for r in self.states:
            if not isinstance(r, MTPRoundState) or r.stage not in ("decode", "prefill"):
                raise ValueError("native decode/prefill round states required")
            if any(not math.isfinite(v) or v <= 0 for v in (r.weight, r.seconds)):
                raise ValueError("positive finite state weight and seconds required")
            if (len(r.decode_contexts) != len(r.decode_query_lens)
                    or any(not math.isfinite(c) or c <= 0 for c in r.decode_contexts)):
                raise ValueError("one positive current context per decoding request required")
            qs = r.decode_query_lens + r.prefill_query_lens
            if not qs or any(not isinstance(q, int) or isinstance(q, bool) or q <= 0 for q in qs):
                raise ValueError("positive integer actual query widths required")
            if any(q > self.k + 1 for q in r.decode_query_lens):
                raise ValueError("decode query exceeds configured speculative width")
            if (r.stage == "decode" and (not r.decode_query_lens or r.prefill_query_lens)
                    or r.stage == "prefill" and not r.prefill_query_lens):
                raise ValueError("state stage does not match native requests")
            if any(not math.isfinite(v) or v < 0 for v in (
                    r.decode_tokens, r.effective_prefill_tokens, r.initial_output_tokens)):
                raise ValueError("finite nonnegative committed tokens/progress required")
            if (r.decode_tokens > sum(r.decode_query_lens)
                    or r.initial_output_tokens > len(r.prefill_query_lens)
                    or r.effective_prefill_tokens > sum(r.prefill_query_lens)):
                raise ValueError("yield/progress exceeds actual native shape")
            if r.stage == "prefill" and r.effective_prefill_tokens <= 0:
                reasons.append("prefill_state_has_no_new_input_progress")
            if (not isinstance(r.preemptions, int) or isinstance(r.preemptions, bool) or r.preemptions < 0
                    or not isinstance(r.recompute, bool)):
                raise ValueError("invalid preemption evidence")
            if r.preemptions or r.recompute:
                reasons.append("preemption_or_recompute_not_covered")
            if sum(qs) > limits.max_num_batched_tokens or len(qs) > limits.max_num_seqs:
                reasons.append("state_exceeds_engine_limits")
            if set(r.required_pages) != set(limits.available_pages):
                reasons.append("state_physical_pool_profile_missing_or_mismatched")
            elif any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in r.required_pages.values()):
                raise ValueError("nonnegative integer physical page requirements required")
            elif any(n > limits.available_pages[pool] for pool, n in r.required_pages.items()):
                reasons.append("state_page_budget_exceeded")
        if not any(r.stage == "decode" for r in self.states):
            reasons.append("missing_pure_decode_distribution")
        if stats.mean_prefill_tokens > 0 and not any(r.stage == "prefill" for r in self.states):
            reasons.append("missing_prefill_distribution")
        if self.workspace_bytes is None:
            reasons.append("workspace_profile_missing")
        elif not isinstance(self.workspace_bytes, int) or isinstance(self.workspace_bytes, bool) or self.workspace_bytes < 0:
            raise ValueError("workspace must be nonnegative integer bytes")
        elif self.workspace_bytes > limits.non_kv_headroom_bytes:
            reasons.append("workspace_budget_exceeded")
        return sorted(set(reasons))


@dataclass(frozen=True)
class MTPRangeProfile:
    """Execution evidence scoped to an INITIAL context interval and policy.

    The interval is not the per-round current context in ``MTPRoundState``.
    Keep model/tokenizer and generation policy identities with the profile;
    never transfer Plan/short/cold evidence to Task/long/cached implicitly.
    """

    agent_name: str
    lower_bound: int
    upper_bound: int
    reuse: bool
    model_id: str
    generation_policy_id: str
    mode: MTPExecutionMode | MTPExecutionDistribution
    limits: MTPResourceLimits

    def __post_init__(self):
        if (not self.agent_name or not self.model_id or not self.generation_policy_id
                or not isinstance(self.reuse, bool)
                or not isinstance(self.lower_bound, int) or isinstance(self.lower_bound, bool)
                or not isinstance(self.upper_bound, int) or isinstance(self.upper_bound, bool)
                or self.lower_bound < 0 or self.upper_bound <= self.lower_bound):
            raise ValueError("explicit agent/range/cache/model/generation profile scope required")
        if (not isinstance(self.mode, (MTPExecutionMode, MTPExecutionDistribution))
                or not isinstance(self.limits, MTPResourceLimits)):
            raise ValueError("native execution mode/distribution and resource limits required")


class ThroughputEstimator:
    """Estimates per-instance throughput using latency models and binary search.

    Given request statistics and hardware constraints, finds the maximum
    sustainable request rate (requests/sec) via iterative binary search
    on the scheduler loop budget.
    """

    def __init__(
        self,
        latency_predictor: LatencyPredictor,
        request_analyzer: RequestAnalyzer,
        chunk_size: int = 512,
        cache_swap_latency: float = 0.1,
        context_step: int = 2000,
        max_context_length: int = 200_000,
    ) -> None:
        self._predictor = latency_predictor
        self._analyzer = request_analyzer
        self._chunk_size = chunk_size
        self._swap_latency = cache_swap_latency
        self._context_step = context_step
        self._max_context_length = max_context_length
        self.last_diagnostics = {}

    @property
    def max_context_idx(self) -> int:
        return self._max_context_length // self._context_step

    # ------------------------------------------------------------------
    # MTP opt-in entry. Legacy AR equations and callers below stay unchanged.
    # ------------------------------------------------------------------
    @staticmethod
    def mtp_one_second_budget(rate, stats: MTPRequestStats,
                              mode: MTPExecutionMode | MTPExecutionDistribution):
        """Evaluate a validated fixed shape/distribution, in seconds and tokens/s.

        Do not clamp a negative remaining time, floor the average round count,
        credit speculative queries as accepted output, or add cache-overlap
        output without an independently profiled execution mode.
        """
        if not math.isfinite(rate) or rate < 0:
            raise ValueError("rate must be finite and nonnegative")
        stats.validate()
        if not math.isfinite(mode.decode_seconds) or mode.decode_seconds <= 0:
            raise ValueError("positive finite decode time required")
        if stats.mean_prefill_tokens and (mode.effective_prefill_tokens_per_round <= 0
                or mode.mixed_seconds is None or mode.mixed_seconds <= 0):
            raise ValueError("prefill demand requires an explicit positive mixed time and progress")
        n_mixed = rate * stats.mean_prefill_tokens / mode.effective_prefill_tokens_per_round if stats.mean_prefill_tokens else 0.
        mixed_time = n_mixed * mode.mixed_seconds if n_mixed else 0.
        cache_time = rate * stats.serial_cache_seconds_per_request
        remaining_time = 1. - mixed_time - cache_time
        demand = rate * stats.remaining_decode_tokens
        if remaining_time < 0:
            n_decode = supply = None
            feasible = False
        else:
            n_decode = remaining_time / mode.decode_seconds
            supply = n_mixed * mode.mixed_decode_tokens_per_round + n_decode * mode.decode_tokens_per_round
            feasible = supply >= demand
        return dict(feasible=feasible, rate_requests_per_second=rate,
            mixed_rounds_per_second=n_mixed, mixed_time_seconds=mixed_time,
            serial_cache_time_seconds=cache_time, remaining_decode_time_seconds=remaining_time,
            decode_rounds_per_second=n_decode, decode_supply_tokens_per_second=supply,
            decode_demand_tokens_per_second=demand)

    @staticmethod
    def analyze_mtp_throughput(stats: MTPRequestStats, modes, limits: MTPResourceLimits,
                               *, absolute_tolerance=1e-7, max_rate=1e6):
        """Candidate execution envelope, NOT a native scheduler/SLO guarantee.

        Keep each candidate fixed during bisection. Reject unsupported profiles
        explicitly instead of extrapolating, copying nearest B/K, or reporting
        unknown as zero capacity. No workflow/invocation normalization here.
        """
        stats.validate()
        limits.validate()
        if (not math.isfinite(absolute_tolerance) or absolute_tolerance <= 0
                or not math.isfinite(max_rate) or max_rate <= absolute_tolerance):
            raise ValueError("invalid bisection limits")
        candidates, names = [], set()
        for mode in modes:
            if mode.name in names:
                raise ValueError("duplicate candidate name")
            names.add(mode.name)
            reasons = mode.unsupported_reasons(stats, limits)
            if reasons:
                candidates.append(dict(name=mode.name, k=mode.k, status="unsupported", reasons=reasons,
                                       capacity_requests_per_second=None))
                continue
            left, right = 0., min(1., max_rate)
            budget = lambda rate: ThroughputEstimator.mtp_one_second_budget(rate, stats, mode)
            while right < max_rate and budget(right)["feasible"]:
                left, right = right, min(right * 2., max_rate)
            if budget(right)["feasible"]:
                candidates.append(dict(name=mode.name, k=mode.k, status="unbracketed", reasons=["capacity_above_search_limit"],
                                       capacity_requests_per_second=None))
                continue
            for _ in range(100):
                if right - left <= absolute_tolerance:
                    break
                mid = (left + right) / 2.
                if budget(mid)["feasible"]:
                    left = mid
                else:
                    right = mid
            candidates.append(dict(name=mode.name, k=mode.k, status="candidate", reasons=[],
                capacity_requests_per_second=left, upper_bound_requests_per_second=right,
                budget_at_capacity=budget(left), provenance=mode.provenance))
        supported = [r for r in candidates if r["status"] == "candidate"]
        best = max(supported, key=lambda r: r["capacity_requests_per_second"]) if supported else None
        return dict(status="candidate" if best else "unsupported",
            capacity_requests_per_second=best["capacity_requests_per_second"] if best else None,
            selected_mode=best["name"] if best else None, candidates=candidates,
            suitable_for_allocator=False,
            scope="fixed_mode_envelope_NOT_native_FCFS_or_stable_arrival_capacity")

    @staticmethod
    def analyze_mtp_distributions(stats: MTPRequestStats, modes, limits: MTPResourceLimits, **kwargs):
        """Use the EXISTING one-second equations and bisection on fixed mixtures.

        Only the input representation and its per-state checks differ. We do
        not optimize state frequencies inside bisection or round a mean B.
        """
        modes = tuple(modes)
        if any(not isinstance(mode, MTPExecutionDistribution) for mode in modes):
            raise ValueError("execution distributions required; do not fabricate fixed mean batches")
        result = ThroughputEstimator.analyze_mtp_throughput(stats, modes, limits, **kwargs)
        result["scope"] = "conditional_empirical_state_distribution_NOT_FCFS_or_stable_arrival_guarantee"
        result["profiling_basis"] = "native_calibration_cost_and_yield_NOT_independent_D_V_formula_validation"
        return result

    def estimate_mtp_capacity_in_range(
        self, lower_bound: int, upper_bound: int, agent_name: str, k: int,
        profiles, *, model_id: str, generation_policy_id: str,
        reuse: bool = True, serial_cache_seconds_per_request: float = 0.0,
    ):
        """Target trace demand + separately profiled ability -> LLM requests/s.

        No calibration p/o substitution, workflow normalization, nearest-range
        extrapolation, profiling refit or live allocator writes. The original
        one-second budget and bisection are reused unchanged. Fixed empirical
        distributions remain conditional, not an FCFS stability guarantee.
        """
        if not isinstance(k, int) or isinstance(k, bool) or k < 0:
            raise ValueError("K must be a nonnegative integer")
        if not model_id or not generation_policy_id:
            raise ValueError("target model/tokenizer and generation policy identities required")
        target = self._analyzer.get_mtp_requests_in_range(
            lower_bound, upper_bound, agent_name, reuse=reuse)
        result = dict(agent=agent_name, initial_context_range=[lower_bound, upper_bound],
            K=k, reuse=reuse, model_id=model_id, generation_policy_id=generation_policy_id,
            workload_statistics=target, status="unsupported", capacity_requests_per_second=None,
            selected_mode=None, candidates=[], reasons=[], suitable_for_allocator=False,
            scope="target_trace_demand_with_fixed_execution_evidence_NOT_FCFS_stability_guarantee",
            normalization="LLM_requests_per_second_NOT_invocations_or_workflows",
            cost_basis="profiled_mode_or_distribution_NOT_an_independent_D_V_validation")
        if target["status"] == "empty":
            result["reasons"] = ["no_target_requests_in_range"]
            return result
        stats = MTPRequestStats(target["mean_prefill_tokens"], target["mean_output_tokens"],
            target["mean_initial_output_tokens"], serial_cache_seconds_per_request)
        stats.validate()
        result["request_stats"] = asdict(stats)
        matching = []
        for p in profiles:
            if not isinstance(p, MTPRangeProfile):
                raise ValueError("explicit MTPRangeProfile entries are required")
            if (p.agent_name == agent_name and p.lower_bound == lower_bound
                    and p.upper_bound == upper_bound and p.reuse == reuse
                    and p.model_id == model_id and p.generation_policy_id == generation_policy_id
                    and p.mode.k == k):
                matching.append(p)
        if not matching:
            result["reasons"] = ["no_execution_profile_for_requested_agent_range_K_cache_model_policy"]
            return result
        names = set()
        for p in matching:
            if p.mode.name in names:
                raise ValueError("duplicate execution profile name for requested scope")
            names.add(p.mode.name)
            solver = (self.analyze_mtp_distributions if isinstance(p.mode, MTPExecutionDistribution)
                      else self.analyze_mtp_throughput)
            result["candidates"].extend(solver(stats, [p.mode], p.limits)["candidates"])
        supported = [c for c in result["candidates"] if c["status"] == "candidate"]
        if supported:
            best = max(supported, key=lambda c: c["capacity_requests_per_second"])
            result.update(status="candidate", selected_mode=best["name"],
                capacity_requests_per_second=best["capacity_requests_per_second"])
        else:
            result["reasons"] = sorted({r for c in result["candidates"] for r in c["reasons"]})
        return result

    # ------------------------------------------------------------------
    # Single-instance throughput (binary search)
    # ------------------------------------------------------------------
    def analyze_throughput(
        self,
        num_layers: int,
        instance_mem_tokens: int,
        request_stats: dict,
        interference: bool = True,
    ) -> float:
        """Estimate max throughput for a single instance with given request pattern.

        Uses binary search to find the maximum request rate such that
        the sum of prefill, decode, and swap times fits in one second.

        Args:
            num_layers: Number of transformer layers.
            instance_mem_tokens: Max tokens the instance's KV cache can hold.
            request_stats: Dict with keys E_decode, E_prompt, E_cached, E_max.
            interference: Whether to account for decode interference.

        Returns:
            Maximum requests per second.
        """
        E_D = request_stats["E_decode"]
        E_P = request_stats["E_prompt"]
        E_C = request_stats["E_cached"]
        E_max = request_stats["E_max"]

        # Memory-limited batch size (reserve one slot for prefill)
        if E_D <= 0 or min(E_C, E_P) < 0:
            raise ValueError("invalid request statistics")
        B_max = int(max(1, instance_mem_tokens // max(1, E_C + E_P + E_D // 2)))
        # Reserve positive prefill space and stay inside the calibrated domain.
        B_max = min(B_max, self._chunk_size - 1,
                    getattr(self._predictor, "max_decode_batch", B_max))
        if B_max < 1:
            raise ValueError("chunk/token budget cannot hold a decoding request")

        left = 0.0
        right = 20.0
        swap_latency = (self._predictor.swap_latency(E_C)
                        if hasattr(self._predictor, "swap_latency") else self._swap_latency)

        # Keep the original feasibility equations, but do not silently treat
        # the historical hard-coded 20 req/s as a measured saturation limit.
        def feasible(mid):

            # Cache swap overhead
            time_swap = min(mid * swap_latency, 1.0)
            if time_swap >= 1.0:
                return False

            # Expected number of prefill chunks
            E_K = (E_P * mid) / self._chunk_size
            B = int(min(B_max, np.ceil(E_D * mid / max(E_K, 1e-9))))
            if B < 1:
                return False
            if hasattr(self._predictor, "supported_decode_batches"):
                B = max((value for value in self._predictor.supported_decode_batches if value <= B), default=0)
                if not B:
                    return False

            # Chunk prefill latency
            L_prefill_chunk = self._predictor.predict_latency(
                num_layers=num_layers,
                decode_batch_size=B,
                avg_decode_length=int(E_C + E_P + E_D // 2),
                longest_decode_length=E_max,
                prefill_query_len=min(E_P, self._chunk_size - B),
                prefill_seq_len=int(E_C + E_P // 2),
                interference=interference,
            )

            time_prefill = min(E_K * L_prefill_chunk, 1.0 - time_swap)
            if time_prefill >= 1.0 - time_swap:
                return False

            # Decode iteration latency
            L_decode = self._predictor.predict_latency(
                num_layers=num_layers,
                decode_batch_size=B,
                avg_decode_length=int(E_C + E_P + E_D // 2),
                longest_decode_length=E_max,
                prefill_query_len=0,
                prefill_seq_len=0,
                interference=interference,
            )

            num_decode_iter = max(0, int(
                (1.0 - time_prefill - time_swap) // max(L_decode, 1e-9)
            ))

            # Tokens per second: decode + prefill + swap contributions
            swap_tokens_contrib = (
                max(0, int(B - mid)) * int(time_swap // max(L_decode, 1e-9))
            )
            prefill_tokens_contrib = (
                max(0, int(B - mid)) * int(time_prefill // max(L_prefill_chunk, 1e-9))
            )
            tokens_per_sec = (
                swap_tokens_contrib
                + prefill_tokens_contrib
                + num_decode_iter * B
            )

            return tokens_per_sec > E_D * mid

        while right < 1e6 and feasible(right):
            left, right = right, right * 2
        if right >= 1e6 and feasible(right):
            raise ValueError("could not bracket AR capacity; inspect the profile/model")
        for _ in range(24):
            mid = (right + left) / 2.0
            if feasible(mid):
                left = mid
            else:
                right = mid
        self.last_diagnostics = {"batch_limit": B_max, "swap_seconds_per_request": swap_latency,
                                 "capacity_requests_per_second": left, "upper_bound": right}
        return left

    # ------------------------------------------------------------------
    # Batch throughput across context ranges
    # ------------------------------------------------------------------
    def attain_instance_throughput(
        self,
        num_layers: int,
        instance_cached_tokens: int,
        agent_name: str,
        interference: bool = True,
        reuse: bool = True,
    ) -> Dict[Tuple[int, int], float]:
        """Compute instance throughput for all context range partitions.

        Iterates over all (prev_context_idx, context_idx) pairs and computes
        the throughput for requests whose context falls within that range.

        Args:
            num_layers: Number of transformer layers.
            instance_cached_tokens: KV cache capacity per instance.
            agent_name: Agent to analyze.
            interference: Account for interference.
            reuse: Enable KV cache reuse.

        Returns:
            Dict mapping (prev_idx, curr_idx) -> throughput (requests/sec).
        """
        max_idx = self.max_context_idx
        cache: Dict[Tuple[int, int], float] = {}

        for prev_idx in range(0, max_idx):
            lower_bound = prev_idx * self._context_step
            for curr_idx in range(prev_idx + 1, max_idx + 1):
                upper_bound = curr_idx * self._context_step
                key = (prev_idx, curr_idx)

                stats = self._analyzer.get_requests_in_range(
                    lower_bound, upper_bound, agent_name, reuse=reuse,
                )
                if stats["avg_decode_length"] == 0:
                    continue

                request_stats = {
                    "E_max": stats["max_length"],
                    "E_prompt": stats["avg_prefill_length"],
                    "E_decode": stats["avg_decode_length"],
                    "E_cached": stats["avg_cached_tokens"],
                }

                base_tp = self.analyze_throughput(
                    num_layers, instance_cached_tokens, request_stats,
                    interference=interference,
                )

                cache[key] = (
                    base_tp
                    * stats["out_proportion"]
                    / max(stats["query_proportion"], 1e-9)
                )

        return cache
