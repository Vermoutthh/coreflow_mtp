"""Request analyzer for extracting statistics from workflow node lists."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class RequestAnalyzer:
    """Analyzes workflow request traces to extract statistical features.

    Provides methods for:
    - Computing per-agent invocation counts
    - Extracting request statistics within context length ranges
    - Caching computed results for repeated queries with the same parameters
    """

    def __init__(self, node_list_path: str) -> None:
        """Initialize with a path to the node_list JSON file.

        Args:
            node_list_path: Path to the node_list.json file.
        """
        self._node_list_path = node_list_path
        self._node_list: Optional[dict] = None

        # Cache structure: (lower, upper, agent, reuse) -> stats
        self._range_cache: Dict[Tuple[int, int, str, bool], dict] = {}

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_node_list(self) -> dict:
        """Lazy-load the node list from file."""
        if self._node_list is None:
            with open(self._node_list_path, "r") as f:
                self._node_list = json.load(f)
        return self._node_list

    @property
    def node_list(self) -> dict:
        return self._load_node_list()

    @property
    def agents(self) -> List[str]:
        """Discover all unique agent names in the node list."""
        agents_set: set[str] = set()
        for nodes in self.node_list.values():
            for node in nodes:
                agents_set.add(node["agent_name"])
        return sorted(agents_set)

    # ------------------------------------------------------------------
    # Invocation statistics
    # ------------------------------------------------------------------
    def avg_num_invocations(self, agents: Optional[List[str]] = None) -> Dict[str, float]:
        """Compute the average number of unique node invocations per agent.

        For each workflow trace, counts unique (node_id, agent) pairs
        then averages across all traces.

        Args:
            agents: List of agent names to analyze. Defaults to all agents.

        Returns:
            Dict mapping agent_name -> average invocation count.
        """
        node_list = self._load_node_list()
        if agents is None:
            agents = self.agents

        result: Dict[str, float] = {}
        for agent in agents:
            invocation_counts: List[int] = []
            for trace_nodes in node_list.values():
                node_ids: set[str] = set()
                for record in trace_nodes:
                    if record["agent_name"] == agent:
                        nid = record["node_id"].split(":")[0]
                        node_ids.add(nid)
                invocation_counts.append(len(node_ids))
            result[agent] = float(np.mean(invocation_counts))
        return result

    # ------------------------------------------------------------------
    # Range-based request statistics
    # ------------------------------------------------------------------
    def get_requests_in_range(
        self,
        lower_bound: int,
        upper_bound: int,
        agent_name: str,
        reuse: bool = True,
        use_cache: bool = True,
    ) -> dict:
        """Extract statistics for requests whose context length falls in [lower, upper).

        Args:
            lower_bound: Minimum context length (inclusive).
            upper_bound: Maximum context length (exclusive).
            agent_name: Filter to this agent's nodes.
            reuse: If True, cached tokens are considered separately from prefill.
                   If False, cached tokens are merged into prefill.
            use_cache: Whether to use and populate the internal cache.

        Returns:
            dict with keys:
                - avg_decode_length
                - avg_prefill_length
                - avg_cached_tokens
                - max_length
                - out_proportion
                - query_proportion
        """
        cache_key = (lower_bound, upper_bound, agent_name, reuse)
        if use_cache and cache_key in self._range_cache:
            return self._range_cache[cache_key]

        node_list = self._load_node_list()
        result = self._compute_range_stats(
            node_list, lower_bound, upper_bound, agent_name, reuse
        )

        if use_cache:
            self._range_cache[cache_key] = result
        return result

    def get_mtp_requests_in_range(
        self, lower_bound: int, upper_bound: int, agent_name: str,
        reuse: bool = True,
    ) -> dict:
        """Use the ORIGINAL range/grouping logic, with unrounded MTP demand.

        Each trace record is one LLM call, not one token, source history or
        workflow. Outputs must already be recorded for this workload/model;
        this method does not predict future response lengths. The initial
        prefill-produced token is credited once, separately from decode yield.
        A trace may specify ``initial_output_tokens`` explicitly (0 or 1).
        """
        if (not isinstance(lower_bound, int) or isinstance(lower_bound, bool)
                or not isinstance(upper_bound, int) or isinstance(upper_bound, bool)
                or lower_bound < 0 or upper_bound <= lower_bound):
            raise ValueError("a nonempty nonnegative [lower, upper) range is required")
        if not isinstance(reuse, bool) or not agent_name:
            raise ValueError("an agent and explicit boolean cache policy are required")
        result = self._compute_range_stats(
            self._load_node_list(), lower_bound, upper_bound, agent_name,
            reuse, precise=True,
        )
        result["node_list_path"] = str(self._node_list_path)
        return result

    @staticmethod
    def _token_counts(node):
        """Length-only legacy traces and explicit semantic IDs are distinct."""
        def count(value):
            return sum(value) if isinstance(value, list) else int(value)
        cached = count(node.get("cached_tokens", 0))
        ids = node.get("input_token_ids", node.get("prompt_token_ids"))
        if ids is not None:
            # Explicit IDs cover the complete context, not an array of lengths.
            full_length = len(ids)
            if cached < 0 or cached > full_length:
                raise ValueError("cached_tokens exceeds semantic input length")
            prefill = full_length - cached
        else:
            prefill = count(node["input_tokens"])
        if "num_output_tokens" in node:
            output = int(node["num_output_tokens"])
        elif "output_token_ids" in node:
            output = len(node["output_token_ids"])
        else:
            output = count(node["output_tokens"])
        return prefill, output, cached

    @staticmethod
    def _compute_range_stats(
        node_list: dict,
        lower_bound: int,
        upper_bound: int,
        agent_name: str,
        reuse: bool,
        precise: bool = False,
    ) -> dict:
        """Core computation for range-based statistics (stateless)."""
        decode_lengths: List[int] = []
        prefill_lengths: List[int] = []
        cached_list: List[int] = []
        max_length = 0
        initial_outputs = []
        selected_workflows, selected_sources = set(), set()

        total_llm_calls = 0
        total_out = 0
        node_ids_in_range: set[str] = set()
        node_ids_all: set[str] = set()

        for workflow_key, trace_nodes in node_list.items():
            # Group nodes by (node_id, repeat)
            agent_nodes: Dict[str, Dict[int, dict]] = defaultdict(dict)
            for node in trace_nodes:
                if node["agent_name"] != agent_name:
                    continue
                nid, repeat_str = node["node_id"].split(":")
                if precise and int(repeat_str) in agent_nodes[nid]:
                    raise ValueError("duplicate LLM call in workload trace")
                agent_nodes[nid][int(repeat_str)] = node

            for nid, repeat_dict in agent_nodes.items():
                unique_nid = f"{workflow_key}_{nid}"
                node_ids_all.add(unique_nid)
                sorted_repeats = sorted(repeat_dict.keys())

                for i, r in enumerate(sorted_repeats):
                    node = repeat_dict[r]
                    input_tokens, output_tokens, cached_tokens = RequestAnalyzer._token_counts(node)

                    context_length = input_tokens + cached_tokens

                    if lower_bound <= context_length < upper_bound:
                        if precise:
                            values = (input_tokens, output_tokens, cached_tokens)
                            if any(v < 0 or int(v) != v for v in values):
                                raise ValueError("trace token counts must be nonnegative integers")
                            if ("output_token_ids" in node
                                    and len(node["output_token_ids"]) != output_tokens):
                                raise ValueError("output length does not match recorded token IDs")
                            prefill = input_tokens + (0 if reuse else cached_tokens)
                            initial = node.get("initial_output_tokens", int(prefill > 0 and output_tokens > 0))
                            if (not isinstance(initial, int) or isinstance(initial, bool)
                                    or initial not in (0, 1) or initial > output_tokens
                                    or (initial and not prefill)):
                                raise ValueError("initial output must be a witnessed 0/1 prefill token")
                            initial_outputs.append(initial)
                            selected_workflows.add(workflow_key)
                            selected_sources.add(node.get("source_id", f"{workflow_key}_{node['node_id']}"))
                        node_ids_in_range.add(f"{workflow_key}_{nid}")
                        total_llm_calls += 1

                        decode_lengths.append(output_tokens)
                        prefill_lengths.append(input_tokens)
                        cached_list.append(cached_tokens)
                        max_length = max(max_length, context_length + output_tokens)

                        # Determine if this node produces an "out" (final output)
                        if i == len(sorted_repeats) - 1:
                            total_out += 1
                        else:
                            next_r = sorted_repeats[i + 1]
                            next_node = repeat_dict[next_r]
                            next_input, _, next_cached = RequestAnalyzer._token_counts(next_node)
                            next_context = next_input + next_cached
                            if not (lower_bound <= next_context < upper_bound):
                                total_out += 1

        if precise:
            count = total_llm_calls
            mean = lambda values: sum(values) / count if count else None
            prefill = [p + (0 if reuse else c) for p, c in zip(prefill_lengths, cached_list)]
            return {
                "status": "observed" if count else "empty",
                "request_count": count,
                "workflow_count": len(selected_workflows),
                "source_count": len(selected_sources),
                "mean_prefill_tokens": mean(prefill),
                "mean_output_tokens": mean(decode_lengths),
                "mean_cached_tokens": mean(cached_list) if reuse else (0.0 if count else None),
                "mean_context_tokens": mean([p + c for p, c in zip(prefill_lengths, cached_list)]),
                "mean_initial_output_tokens": mean(initial_outputs),
                "max_length": max_length if count else None,
                "out_proportion": total_out / count if count else None,
                "query_proportion": len(node_ids_in_range) / max(len(node_ids_all), 1),
                "weighting": "one_trace_record_one_LLM_call; no_integer_rounding",
            }

        if total_llm_calls == 0:
            return {
                "avg_decode_length": 0,
                "avg_prefill_length": 0,
                "avg_cached_tokens": 0,
                "max_length": 0,
                "out_proportion": 0.0,
                "query_proportion": 0.0,
            }

        sum_decode = sum(decode_lengths)
        sum_prefill = sum(prefill_lengths)
        sum_cached = sum(cached_list)

        avg_decode = int(sum_decode // total_llm_calls)
        if reuse:
            avg_prefill = int(sum_prefill // total_llm_calls)
            avg_cached = int(sum_cached // total_llm_calls)
        else:
            # Match legacy: two separate integer divisions, then add
            avg_prefill = int(sum_prefill // total_llm_calls) + int(sum_cached // total_llm_calls)
            avg_cached = 0

        return {
            "avg_decode_length": avg_decode,
            "avg_prefill_length": avg_prefill,
            "avg_cached_tokens": avg_cached,
            "max_length": max_length,
            "out_proportion": total_out / total_llm_calls,
            "query_proportion": len(node_ids_in_range) / max(len(node_ids_all), 1),
        }

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------
    def clear_cache(self) -> None:
        """Clear the internal range query cache."""
        self._range_cache.clear()

    def cache_size(self) -> int:
        """Return the number of cached range queries."""
        return len(self._range_cache)
