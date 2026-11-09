"""Explicit real-cache leases and transactions in the V1 EngineCore process.

Pinned blocks have real refcounts. Only native APC-validated finalized blocks
are reusable; speculative tails and unhashed GDN running states are excluded.
An added EngineCore utility method executes transactions on its own main loop,
not from an unsafe HTTP thread. No site-packages file is edited.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time
import uuid

from vllm.v1.core.sched.scheduler import Scheduler


class CacheScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cf_settings = self.vllm_config.additional_config["coreflow_cache"]
        self.cf_entries = {}
        self.cf_engine = None
        self.cf_reports = {}
        self.cf_layout = []
        self.cf_swap_samples = {"d2h": [], "h2d": []}
        self.cf_step_stats = {"count": 0, "sum": 0,
                              "buckets": {str(b): 0 for b in (1, 2, 4, 8, 16, 32, 64, 128, 256)}}
        self.cf_counters = {"retained": 0, "offloaded": 0, "restored": 0,
                            "imported": 0, "released": 0, "pressure_offloads": 0, "keepalive_expired": 0}
        self.cf_max_blocks = max(1, int(self.kv_cache_config.num_blocks * self.cf_settings["gpu_fraction"]))
        # The scheduler import happens inside EngineCore.__init__. Adding one
        # named, version-pinned utility preserves main-loop serialization.
        from vllm.v1.engine.core import EngineCore
        EngineCore.coreflow_cache = _engine_cache_command

    def _cf_key(self, query_id, invocation_id):
        return json.dumps([str(query_id), str(invocation_id)], separators=(",", ":"))

    def _cf_worker(self, action, entry, artifact_dir=None):
        if self.cf_engine is None:
            raise RuntimeError("cache bridge has not been initialized")
        return self.cf_engine.collective_rpc("coreflow_tensor_cache", timeout=120,
            args=(action, entry["entry_id"], entry["groups"], artifact_dir))

    def _cf_unique_blocks(self, entry):
        pool = self.kv_cache_manager.block_pool
        ids = {block["block_id"] for g in entry["groups"] for block in g["blocks"]}
        return [pool.blocks[block_id] for block_id in sorted(ids)]

    def _cf_drop_gpu(self, entry, evict=False):
        if entry["location"] != "gpu":
            return
        pool = self.kv_cache_manager.block_pool
        blocks = self._cf_unique_blocks(entry)
        pool.free_blocks(blocks)
        if evict:
            for block in blocks:
                if block.ref_cnt == 0:
                    pool._maybe_evict_cached_block(block)

    def _cf_release(self, key):
        entry = self.cf_entries.pop(key, None)
        if entry is not None:
            self._cf_drop_gpu(entry, evict=True)
            self._cf_worker("drop", entry)
            self.cf_counters["released"] += 1

    def _cf_offload(self, entry):
        if entry["location"] == "cpu":
            return entry
        started = time.perf_counter()
        # The copy finishes on every TP rank before GPU refs are released.
        while True:
            try:
                evidence = self._cf_worker("snapshot", entry)
                break
            except Exception as exc:
                if "CPU cache" not in str(exc):
                    raise
                victims = [v for v in self.cf_entries.values() if v is not entry and v["location"] == "cpu"]
                if not victims:
                    raise
                victim = min(victims, key=lambda value: value["created"])
                self._cf_release(self._cf_key(victim["query_id"], victim["invocation_id"]))
        self._cf_drop_gpu(entry, evict=True)
        entry["location"] = "cpu"
        entry["copy_evidence"] = evidence
        self.cf_counters["offloaded"] += 1
        self._cf_record_swap("d2h", evidence, time.perf_counter() - started)
        return entry

    def _cf_record_swap(self, direction, evidence, seconds):
        total = sum(rank.get("bytes", 0) for rank in evidence)
        if total <= 0:
            return
        if not hasattr(self, "cf_swap_samples"):
            self.cf_swap_samples = {"d2h": [], "h2d": []}
        values = self.cf_swap_samples[direction]
        values.append({"bytes": total, "seconds": seconds})
        del values[:-128]

    def _cf_swap_model(self):
        from coreflow_mtp.predictor.linear import LinearRegression
        result = {"provenance": {"scope": "native_cache_transactions_including_RPC_hashing_verification"}}
        for direction, values in self.cf_swap_samples.items():
            result["provenance"]["samples_" + direction] = len(values)
            sizes = {r["bytes"] for r in values}
            if not sizes:
                return {"available": False, **result}
            if len(sizes) == 1:
                import numpy as np
                result[direction] = {"intercept_seconds": float(np.median([r["seconds"] for r in values])),
                    "seconds_per_byte": 0., "valid_bytes": [min(sizes), max(sizes)],
                    "fit_kind": "measured_fixed_size_not_general_linear_fit"}
                continue
            model = LinearRegression().fit([[r["bytes"]] for r in values], [r["seconds"] for r in values])
            result[direction] = {"intercept_seconds": max(0., model.intercept_),
                "seconds_per_byte": max(0., float(model.coef_[0])),
                "valid_bytes": [min(sizes), max(sizes)], "fit_kind": "measured_linear"}
        return {"available": True, **result}

    def _free_blocks(self, request):
        policy = (request.sampling_params.extra_args or {}).get("coreflow_cache") if request.sampling_params else None
        if policy:
            key = self._cf_key(policy["query_id"], policy["invocation_id"])
            report_id = policy.get("engine_request_id", request.request_id)
            # Aborted/errored requests never create successful cache leases.
            from vllm.v1.request import RequestStatus
            if policy.get("free_cache", True):
                self._cf_release(key)
                self.cf_reports[report_id] = {"stored": False, "released": True}
            elif request.status in (RequestStatus.FINISHED_STOPPED, RequestStatus.FINISHED_LENGTH_CAPPED):
                blocks, reusable = self.kv_cache_manager.get_computed_blocks(request)
                owned = self.kv_cache_manager.get_blocks(request.request_id)
                groups = []
                finalized = min(request.num_computed_tokens, request.num_tokens - 1)
                for gid, hit_group in enumerate(blocks.blocks):
                    size = self.kv_cache_config.kv_cache_groups[gid].kv_cache_spec.block_size
                    # MTP/EAGLE native lookup intentionally drops its last
                    # matched full-attention block. Transferring only the
                    # already-truncated hit would cause a second drop in the
                    # destination and lose all hits at the first boundary.
                    # Retain finalized hashed guard pages too, without ever
                    # including speculative/uncomputed tails.
                    merged = {pos: block for pos, block in enumerate(owned.blocks[gid])
                              if (pos + 1) * size <= finalized and
                              not block.is_null and block.block_hash is not None}
                    merged.update({pos: block for pos, block in enumerate(hit_group)
                                   if not block.is_null and block.block_hash is not None})
                    groups.append({"gid": gid, "blocks": [
                        {"position": pos, "block_id": block.block_id}
                        for pos, block in sorted(merged.items())]})
                if reusable and all(g["blocks"] for g in groups):
                    entry = {"entry_id": uuid.uuid4().hex, "query_id": policy["query_id"],
                             "invocation_id": policy["invocation_id"], "location": "gpu",
                             "groups": groups, "token_ids": list(request.all_token_ids),
                             "reusable_tokens": reusable, "created": time.time(),
                             "keep_cache_in_gpu": policy.get("keep_cache_in_gpu", False)}
                    adaptive = policy.get("adaptive_policy")
                    decision = None
                    if adaptive is not None:
                        from coreflow_mtp.cache.policy import decide_from_bytes
                        if not self.cf_layout:
                            raise RuntimeError("predictive policy requires validated actual worker page layouts")
                        nbytes = sum(len(group["blocks"]) * rank["specs"][group["gid"]]["page_size"]
                                     * len(rank["specs"][group["gid"]]["storages"])
                                     for rank in self.cf_layout for group in groups)
                        decision = decide_from_bytes(adaptive, nbytes)
                        entry["keep_cache_in_gpu"] = decision["keep_cache_in_gpu"]
                        if entry["keep_cache_in_gpu"]:
                            entry["keepalive_until"] = time.monotonic() + decision["retain_seconds"]
                    hard_ids = {b.block_id for k, other in self.cf_entries.items()
                                if k != key and other["location"] == "gpu" and other["keep_cache_in_gpu"]
                                for b in self._cf_unique_blocks(other)}
                    hard_ids.update(b.block_id for b in self._cf_unique_blocks(entry))
                    if adaptive is not None and entry["keep_cache_in_gpu"] and len(hard_ids) > self.cf_max_blocks:
                        entry["keep_cache_in_gpu"] = False
                        entry.pop("keepalive_until", None)
                        decision.update(keep_cache_in_gpu=False, retain_seconds=0., reason="gpu_budget_cpu_fallback")
                    if entry["keep_cache_in_gpu"] and len(hard_ids) > self.cf_max_blocks:
                        self.cf_reports[report_id] = {"stored": False,
                            "reason": "gpu_residency_budget_exceeded", "previous_owner_preserved": key in self.cf_entries}
                    else:
                        # Hold new references before replacing a shared old owner.
                        self.kv_cache_manager.block_pool.touch(self._cf_unique_blocks(entry))
                        self._cf_release(key)
                        self.cf_entries[key] = entry
                        self.cf_counters["retained"] += 1
                        self.cf_reports[report_id] = {"stored": True,
                            "reusable_tokens": reusable, "location": "gpu",
                            "keep_cache_in_gpu": entry["keep_cache_in_gpu"],
                            **({"policy": decision} if decision is not None else {})}
                else:
                    self._cf_release(key)
                    self.cf_reports[report_id] = {"stored": False,
                        "reason": "no_complete_native_hybrid_prefix"}
            if len(self.cf_reports) > 1024:
                self.cf_reports.pop(next(iter(self.cf_reports)))
        super()._free_blocks(request)

    def _cf_restore(self, entry):
        if entry["location"] == "gpu":
            return entry
        started = time.perf_counter()
        from vllm import SamplingParams
        from vllm.v1.request import Request
        from vllm.v1.core.kv_cache_utils import BlockHashListWithBlockSize, make_block_hash_with_group_id
        pool = self.kv_cache_manager.block_pool
        count = sum(len(g["blocks"]) for g in entry["groups"])
        hard_ids = {b.block_id for other in self.cf_entries.values()
                    if other is not entry and other["location"] == "gpu" and other["keep_cache_in_gpu"]
                    for b in self._cf_unique_blocks(other)}
        if count + len(hard_ids) > self.cf_max_blocks:
            raise MemoryError("entry exceeds the explicit GPU residency budget")
        if pool.get_num_free_blocks() < count:
            for other in sorted(self.cf_entries.values(), key=lambda e: e["created"]):
                if other is not entry and other["location"] == "gpu" and not other["keep_cache_in_gpu"]:
                    self._cf_offload(other)
                if pool.get_num_free_blocks() >= count:
                    break
        fresh = pool.get_new_blocks(count)
        cursor = 0
        plan = copy.deepcopy(entry)
        for group in plan["groups"]:
            for block in group["blocks"]:
                block["block_id"] = fresh[cursor].block_id
                cursor += 1
        try:
            evidence = self._cf_worker("restore", plan)
            # Hashes are recomputed by the destination's actual V1 hasher.
            # Source processes may have independent random NONE_HASH seeds.
            req = Request("coreflow-import-hash", list(entry["token_ids"]),
                SamplingParams(max_tokens=1), None, block_hasher=self.cf_engine.request_block_hasher)
            for group in plan["groups"]:
                gid = group["gid"]
                size = self.kv_cache_config.kv_cache_groups[gid].kv_cache_spec.block_size
                hashes = (req.block_hashes if size == pool.hash_block_size else
                          BlockHashListWithBlockSize(req.block_hashes, pool.hash_block_size, size))
                for block in group["blocks"]:
                    value = make_block_hash_with_group_id(hashes[block["position"]], gid)
                    physical = pool.blocks[block["block_id"]]
                    physical.block_hash = value
                    pool.cached_block_hash_to_block.insert(value, physical)
            entry["groups"] = plan["groups"]
            entry["location"] = "gpu"
            entry["copy_evidence"] = evidence
            # CPU data is not silently kept after an explicit swap-in.
            self._cf_worker("drop", entry)
            self.cf_counters["restored"] += 1
            self._cf_record_swap("h2d", evidence, time.perf_counter() - started)
            return entry
        except BaseException:
            for block in fresh:
                if block.block_hash is not None:
                    pool._maybe_evict_cached_block(block)
            pool.free_blocks(fresh)
            raise

    def schedule(self):
        self._cf_mark_expired()
        # Preserve forward progress when pinned owners otherwise exhaust the
        # native pool. At this synchronous boundary no model step is in flight.
        pool = self.kv_cache_manager.block_pool
        reserve = min(pool.num_gpu_blocks // 3, max(2, self.max_model_len // self.block_size))
        gpu = sorted((e for e in self.cf_entries.values() if e["location"] == "gpu" and not e["keep_cache_in_gpu"]), key=lambda e: e["created"])
        held = sum(len(self._cf_unique_blocks(e)) for e in gpu)
        for entry in gpu:
            if pool.get_num_free_blocks() >= reserve and held <= self.cf_max_blocks:
                break
            try:
                count = len(self._cf_unique_blocks(entry))
                self._cf_offload(entry)
                held -= count
                self.cf_counters["pressure_offloads"] += 1
            except MemoryError:
                # Bounded cache follows eviction semantics when both tiers are
                # full; metadata is removed, never advertised as still resident.
                self._cf_release(self._cf_key(entry["query_id"], entry["invocation_id"]))
        output = super().schedule()
        count = len(output.num_scheduled_tokens)
        if count:
            self.cf_step_stats["count"] += 1
            self.cf_step_stats["sum"] += count
            for bucket in self.cf_step_stats["buckets"]:
                self.cf_step_stats["buckets"][bucket] += count <= int(bucket)
        return output

    def _cf_mark_expired(self):
        for entry in self.cf_entries.values():
            until = entry.get("keepalive_until")
            if entry["location"] == "gpu" and until is not None and until <= time.monotonic():
                entry.pop("keepalive_until", None)
                entry["keep_cache_in_gpu"] = False
                entry["expired_keepalive"] = True
                self.cf_counters["keepalive_expired"] += 1

    def cf_command(self, core, payload):
        self.cf_engine = core
        op = payload["op"]
        if op == "validate":
            self.cf_layout = self._cf_worker("layout", {"entry_id": "", "groups": []})
            return {"workers": self.cf_layout}
        if op == "swap_profile":
            return {**self._cf_swap_model(), "observations": copy.deepcopy(self.cf_swap_samples)}
        if op == "expire":
            self._cf_mark_expired()
            if self.get_num_unfinished_requests():
                return {"deferred": True}
            done = 0
            for entry in list(self.cf_entries.values()):
                if entry["location"] == "gpu" and entry.get("expired_keepalive", False):
                    self._cf_offload(entry)
                    entry.pop("expired_keepalive", None)
                    done += 1
            return {"offloaded": done}
        if op == "report":
            return self.cf_reports.pop(payload["engine_request_id"], {"stored": False, "reason": "no_cache_policy"})
        if op == "lookup":
            from vllm import SamplingParams
            from vllm.v1.request import Request
            tokens = payload["token_ids"]
            req = Request("coreflow-lookup", list(tokens), SamplingParams(max_tokens=1), None,
                          block_hasher=core.request_block_hasher)
            _, hits = self.kv_cache_manager.coordinator.find_longest_cache_hit(req.block_hashes, len(tokens) - 1)
            cpu = max((value["reusable_tokens"] for value in self.cf_entries.values()
                       if value["location"] == "cpu" and len(tokens) > value["reusable_tokens"] > 0
                       and tokens[:value["reusable_tokens"]] == value["token_ids"][:value["reusable_tokens"]]), default=0)
            return {"gpu_tokens": hits, "cpu_tokens": cpu, "evidence": "native_v1_lookup_and_real_cpu_directory"}
        if op == "state":
            pool = self.kv_cache_manager.block_pool
            values = []
            for entry in self.cf_entries.values():
                record = copy.deepcopy(entry)
                if entry["location"] == "gpu":
                    record["ref_counts"] = [b.ref_cnt for b in self._cf_unique_blocks(entry)]
                values.append(record)
            return {"entries": values, "free_blocks": pool.get_num_free_blocks(),
                    "total_blocks": pool.num_gpu_blocks, "gpu_budget_blocks": self.cf_max_blocks,
                    "counters": dict(self.cf_counters), "hash_protocol": "vllm-v1-native",
                    "step_stats": copy.deepcopy(self.cf_step_stats),
                    "swap_cost_model": self._cf_swap_model(),
                    "model": self.vllm_config.model_config.model,
                    "tensor_parallel_size": self.parallel_config.tensor_parallel_size,
                    "speculative_k": self.num_spec_tokens,
                    "cpu_cache_workers": self._cf_worker("host_state", {"entry_id": "", "groups": []})}
        key = self._cf_key(payload["query_id"], payload["invocation_id"])
        entry = self.cf_entries.get(key)
        if op == "prepare":
            tokens = payload["token_ids"]
            matches = [value for value in self.cf_entries.values()
                       if len(tokens) > value["reusable_tokens"] > 0 and
                       tokens[:value["reusable_tokens"]] == value["token_ids"][:value["reusable_tokens"]]]
            if not matches:
                return {"found": False, "reason": "no_computed_matching_prefix"}
            entry = max(matches, key=lambda value: (value["reusable_tokens"], value["location"] == "gpu"))
            n = entry["reusable_tokens"]
            entry.pop("keepalive_until", None)
            entry.pop("expired_keepalive", None)
            if entry["location"] == "cpu":
                if self.get_num_unfinished_requests():
                    return {"deferred": True}
                try:
                    self._cf_restore(entry)
                except MemoryError:
                    return {"found": False, "reason": "gpu_budget_recompute"}
            return {"found": True, "reusable_tokens": n}
        if op == "release":
            self._cf_release(key)
            return {"released": entry is not None}
        if self.get_num_unfinished_requests():
            if payload.get("wait_for_idle"):
                return {"deferred": True}
            raise RuntimeError("cache maintenance requires a quiescent instance; retry after current requests finish")
        if op in {"offload", "restore", "export"} and entry is None:
            return {"found": False}
        if op == "offload":
            return {"found": True, "entry": self._cf_offload(entry)}
        if op == "restore":
            self._cf_restore(entry)
            entry["keep_cache_in_gpu"] = True
            entry.pop("keepalive_until", None)
            entry.pop("expired_keepalive", None)
            return {"found": True, "entry": entry}
        if op == "export":
            directory = Path(self.cf_settings["storage_root"]) / uuid.uuid4().hex
            if entry["location"] == "gpu":
                self._cf_worker("snapshot", entry)
            evidence = self._cf_worker("save_artifact", entry, str(directory))
            if entry["location"] == "gpu":
                self._cf_worker("drop", entry)
            manifest = copy.deepcopy(entry)
            manifest.update(format_version=1, model=self.vllm_config.model_config.model,
                            speculative_k=self.num_spec_tokens,
                            tensor_parallel_size=self.parallel_config.tensor_parallel_size,
                            compatibility_id=self.cf_settings["compatibility_id"],
                            copy_evidence=evidence, artifact_dir=str(directory))
            return manifest
        if op == "import":
            manifest = payload["manifest"]
            if (manifest.get("format_version") != 1 or
                manifest.get("compatibility_id") != self.cf_settings["compatibility_id"] or
                manifest["model"] != self.vllm_config.model_config.model or
                manifest["speculative_k"] != self.num_spec_tokens or
                manifest["tensor_parallel_size"] != self.parallel_config.tensor_parallel_size):
                raise ValueError("migration requires compatible model, K, TP and format")
            groups = manifest["groups"]
            if [g["gid"] for g in groups] != list(range(len(self.kv_cache_config.kv_cache_groups))):
                raise ValueError("migration must contain every KV/GDN/MTP group exactly once")
            tokens = manifest["token_ids"]
            if (not isinstance(tokens, list) or len(tokens) > self.max_model_len or
                any(type(token) is not int or token < 0 for token in tokens) or
                not 0 < manifest["reusable_tokens"] < len(tokens)):
                raise ValueError("invalid migration token metadata")
            for group in groups:
                positions = [b["position"] for b in group["blocks"]]
                size = self.kv_cache_config.kv_cache_groups[group["gid"]].kv_cache_spec.block_size
                if (not positions or any(type(p) is not int or p < 0 or (p + 1) * size > len(tokens)
                    for p in positions) or positions != sorted(set(positions))):
                    raise ValueError("invalid or duplicate migration block positions")
            imported = copy.deepcopy(manifest)
            imported.update(entry_id=uuid.uuid4().hex, query_id=payload["query_id"],
                            invocation_id=payload["invocation_id"], location="cpu", created=time.time(),
                            keep_cache_in_gpu=payload.get("keep_cache_in_gpu", True))
            try:
                self._cf_worker("load_artifact", imported, manifest["artifact_dir"])
                if payload.get("keep_cache_in_gpu", True):
                    self._cf_restore(imported)
                self._cf_release(key)
                self.cf_entries[key] = imported
                self.cf_counters["imported"] += 1
                return {"imported": True, "entry": imported}
            except BaseException:
                self._cf_drop_gpu(imported, evict=True)
                self._cf_worker("drop", imported)
                raise
        raise ValueError(f"unknown cache operation {op}")


def _engine_cache_command(core, payload: dict) -> dict:
    if not isinstance(core.scheduler, CacheScheduler):
        raise RuntimeError("CoreFlow cache scheduler is not installed")
    return core.scheduler.cf_command(core, payload)
