"""Context/load routing with optional, explicitly advertised cache support."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any

import httpx

from coreflow_mtp.instance_manager import InstanceInfo, InstanceManager

logger = logging.getLogger(__name__)


class AsyncAgentScheduler:
    def __init__(self, agent_id: str, instance_manager: InstanceManager, http_client,
                 routing_policy="strict"):
        if routing_policy not in {"strict", "coreflow"}:
            raise ValueError("routing policy must be strict or coreflow")
        self.routing_policy = routing_policy
        self.agent_id = agent_id
        self._im = instance_manager
        self._http = http_client
        self._instance_cache_states: dict[str, dict[str, Any]] = {}

    async def _refresh(self, info: InstanceInfo):
        # Original CoreFlow refreshes /cache_state once per instance. It
        # already carries active_requests; do not precede it with /status.
        # Instances without explicit cache support still use /status.
        has_cache_state = info.capabilities.get("cache_state", False)
        route = "/cache_state" if has_cache_state else "/status"
        started = time.monotonic()
        for attempt in range(1, 4):
            try:
                response = await self._http.get(info.endpoint + route)
                response.raise_for_status()
                state = response.json()
                info.active_requests = max(0, int(state.get("active_requests", 0)))
                if (state.get("ready") is False or state.get("available") is False
                        or state.get("status") in {"unavailable", "error", "starting", "not_ready"}):
                    return None
                break
            except Exception as exc:
                # Only retry a read-only state GET after connection failures.
                # Never reuse stale cache/load state, retry a generation POST,
                # or multiply the existing long timeout on a blocked engine.
                retry = attempt < 3 and isinstance(exc, (
                    httpx.ConnectError, httpx.ReadError, httpx.WriteError,
                    httpx.RemoteProtocolError,
                ))
                logger.warning(
                    "Routing state unavailable for %s (%s%s), attempt=%d/3 elapsed=%.3fs retry=%s: %r",
                    info.instance_id, info.endpoint, route, attempt,
                    time.monotonic() - started, retry, exc, exc_info=True)
                if not retry:
                    return None
                await asyncio.sleep(.1 * attempt)
        if attempt > 1:
            logger.warning("Routing state recovered for %s (%s%s) on attempt=%d/3 elapsed=%.3fs",
                           info.instance_id, info.endpoint, route, attempt, time.monotonic() - started)
        cache = state if has_cache_state else {}
        self._instance_cache_states[info.instance_id] = cache
        return info, cache

    @staticmethod
    def _compute_request_block_hashes(tokens, block_size, none_hash):
        hashes = []
        previous = none_hash
        for start in range(0, len(tokens) - block_size + 1, block_size):
            previous = int.from_bytes(hashlib.sha256(repr((
                previous == none_hash, previous,
                tuple(tokens[start:start + block_size]), None,
            )).encode()).digest()[:8], "big", signed=True)
            hashes.append(previous)
        return hashes

    @staticmethod
    def _count_hash_hits(hashes, state):
        blocks = state.get("hash_blocks") or {}
        cached = set(map(int, blocks.get("gpu", []))) | set(map(int, blocks.get("cpu", [])))
        hits = 0
        for value in hashes:
            if value not in cached:
                break
            hits += 1
        return hits

    async def route(self, query_id, invocation_id, input_tokens, num_input_tokens):
        infos = self._im.get_instances_for_agent(self.agent_id)
        candidates = [item for item in await asyncio.gather(*(
            self._refresh(info) for info in infos)) if item is not None]
        if not candidates:
            return None
        # Context partitions remain eligibility constraints, not guessed cache hits.
        matching = [(info, cache) for info, cache in candidates
                    if info.context_range is None or
                    info.context_range[0] <= num_input_tokens < info.context_range[1]]
        if not matching and self.routing_policy == "strict":
            return None
        async def native_lookup(item):
            info, state = item
            if state.get("hash_protocol") == "vllm-v1-native":
                try:
                    response = await self._http.post(info.endpoint + "/cache/lookup",
                        json={"prompt_token_ids": input_tokens})
                    response.raise_for_status()
                    state["native_lookup"] = response.json()
                except Exception as exc:
                    # Losing a routing hint never fabricates cache contents or
                    # prevents native execution from recomputing a safe prompt.
                    state["native_lookup"] = {}
                    logger.warning("Native cache lookup failed for %s: %r", info.instance_id, exc)
            return info, state
        matching = await asyncio.gather(*(native_lookup(item) for item in (
            candidates if self.routing_policy == "coreflow" else matching)))

        def has_native_affinity(info, state):
            hit = state.get("native_lookup", {})
            return (info.capabilities.get("cache_affinity_routing")
                    and max(int(hit.get("gpu_tokens", 0)), int(hit.get("cpu_tokens", 0))) > 0)

        matching = [(info, state) for info, state in matching
                    if info.context_range is None or info.context_range[0] <= num_input_tokens < info.context_range[1]
                    or has_native_affinity(info, state)]
        if not matching:
            return None

        def score(item):
            info, state = item
            exact = False
            if info.capabilities.get("retained_gpu_entries"):
                exact = any(str(entry.get("query_id")) == str(query_id) and
                            str(entry.get("invocation_id")) == str(invocation_id) and
                            len(input_tokens) > entry.get("reusable_tokens", 0) > 0 and
                            input_tokens[:entry["reusable_tokens"]] == entry.get("token_ids", [])[:entry["reusable_tokens"]]
                            for entry in state.get("gpu_entries", []))
            hits = 0
            # These tokens describe actual pinned/offloaded native blocks,
            # never a synthetic cached_tokens declaration or guessed V1 hash.
            cpu_exact = False
            gpu_hint = 0
            cpu_hint = 0
            if state.get("hash_protocol") == "vllm-v1-native":
                for entry in state.get("gpu_entries", []) + state.get("cpu_entries", []):
                    n = int(entry.get("reusable_tokens", 0))
                    if n and input_tokens[:n] == entry.get("token_ids", [])[:n] and len(input_tokens) > n:
                        hits = max(hits, n)
                        if entry.get("location") == "gpu":
                            gpu_hint = max(gpu_hint, n)
                        else:
                            cpu_hint = max(cpu_hint, n)
                        cpu_exact |= (entry.get("location") == "cpu" and
                            str(entry.get("query_id")) == str(query_id) and
                            str(entry.get("invocation_id")) == str(invocation_id))
            # Old hash construction is only used when the backend explicitly
            # identifies the old hash protocol. Native V1 hashes are not guessed.
            if (info.capabilities.get("prefix_hashes") and
                    state.get("hash_protocol") == "coreflow-v0-sha256" and
                    state.get("none_hash") is not None):
                block_size = int(state.get("block_size", 0))
                if block_size > 0:
                    hits = self._count_hash_hits(self._compute_request_block_hashes(
                        input_tokens, block_size, state["none_hash"]), state)
            native = state.get("native_lookup", {})
            if self.routing_policy == "coreflow":
                in_range = info.context_range is None or info.context_range[0] <= num_input_tokens < info.context_range[1]
                return (-int(exact), -max(hits, gpu_hint, cpu_hint,
                    int(native.get("gpu_tokens", 0)), int(native.get("cpu_tokens", 0))), -int(in_range),
                    max(info.active_requests, info.local_inflight) + len(state.get("gpu_entries", [])), info.instance_id)
            return (-int(exact), -max(gpu_hint, int(native.get("gpu_tokens", 0))), -int(cpu_exact),
                    -max(cpu_hint, hits, int(native.get("cpu_tokens", 0))),
                    max(info.active_requests, info.local_inflight) + len(state.get("gpu_entries", [])), info.instance_id)

        return min(matching, key=score)[0].instance_id

    async def dispatch(self, instance_id: str, node_dict: dict):
        info = self._im.get(instance_id)
        if info is None or not info.is_ready:
            return None
        info.local_inflight += 1
        try:
            response = await self._http.post(f"{info.endpoint}/generate", json=node_dict)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            # ReadError('') has an empty str(). Preserve its type with ordinary
            # logging, rather than a custom HTTP client or a hidden POST retry.
            logger.warning("dispatch failed for %s: %r", instance_id, exc)
            return {"success": False, "error": f"{type(exc).__name__}: {exc!r}"}
        finally:
            info.local_inflight -= 1

    async def maybe_migrate_request(self, source_instance_id=None, **kwargs):
        source = self._im.get(source_instance_id)
        if source is None or source.context_range is None or not source.capabilities.get("cache_migration"):
            return None
        length = len(kwargs["input_token_ids"]) + len(kwargs["output_token_ids"])
        if length < source.context_range[1]:
            return None
        candidates = [info for info in self._im.get_instances_for_agent(self.agent_id)
            if info.instance_id != source_instance_id and info.capabilities.get("cache_migration")
            and info.context_range is not None and info.context_range[0] <= length < info.context_range[1]
            and info.speculative_tokens == source.speculative_tokens and info.model == source.model]
        if not candidates:
            return {"migrated": False, "reason": "no_compatible_destination"}
        target = min(candidates, key=lambda info: (max(info.active_requests, info.local_inflight), info.instance_id))
        try:
            response = await self._http.post(target.endpoint + "/migrate_cache", json={
                "source_url": source.endpoint, "query_id": kwargs["query_id"],
                "invocation_id": kwargs["invocation_id"], "move": True,
                "keep_cache_in_gpu": kwargs["keep_cache_in_gpu"]})
            response.raise_for_status()
            result = response.json()
            self._instance_cache_states.pop(source_instance_id, None)
            self._instance_cache_states.pop(target.instance_id, None)
            return {"migrated": bool(result.get("success")), "source_instance_id": source_instance_id,
                    "target_instance_id": target.instance_id, "result": result}
        except Exception as exc:
            # A failed transfer does not silently free the source lease or
            # corrupt an already correct generation; expose it in the result.
            return {"migrated": False, "reason": str(exc), "source_preserved": True}

    async def release_request_cache(self, query_id, invocation_id, exclude_instance_id=None):
        results = {}
        for info in self._im.get_instances_for_agent(self.agent_id):
            if info.instance_id == exclude_instance_id or not info.capabilities.get("cache_state"):
                continue
            try:
                response = await self._http.post(info.endpoint + "/cache/release",
                    json={"query_id": query_id, "invocation_id": invocation_id})
                response.raise_for_status()
                results[info.instance_id] = response.json()
            except Exception as exc:
                results[info.instance_id] = {"released": False, "error": str(exc)}
        return results
