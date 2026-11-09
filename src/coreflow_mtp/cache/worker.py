"""Actual paged KV/GDN byte copies on each model worker.

Copy allocator pages, NOT attention kernel tiles. Hybrid attention strides and
Mamba state views share raw page storage; obtaining storage from the bound
layer preserves the allocator's physical page mapping, including MTP layers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from coreflow_mtp.cache.pages import PageStorage


def digest_tensor(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


class CacheWorkerExtension:
    def coreflow_tensor_cache(self, action: str, entry_id: str, groups: list,
                              artifact_dir: str | None = None) -> dict:
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank

        runner = self.model_runner
        config = runner.kv_cache_config
        settings = self.vllm_config.additional_config["coreflow_cache"]
        root = Path(settings["storage_root"]).resolve()
        rank = get_tensor_model_parallel_rank()
        if settings.get("execution_trace"):
            from coreflow_mtp.runtime.tracing import install_worker_trace
            install_worker_trace(self, settings["execution_trace"], rank)
        if not hasattr(self, "_coreflow_host_entries"):
            self._coreflow_host_entries = {}
        host = self._coreflow_host_entries

        def file_path():
            directory = Path(artifact_dir).resolve()
            if not directory.is_relative_to(root):
                raise ValueError("cache artifact must be under the configured shared storage root")
            return directory / f"rank_{rank}.pt"

        if action == "drop":
            host.pop(entry_id, None)
            return {"rank": rank, "dropped": True}
        if action == "host_state":
            return {"rank": rank, "entries": len(host),
                    "bytes": sum(v["bytes"] for v in host.values())}

        # Build a stable layer-storage name and raw-byte view for each group.
        contexts = runner.compilation_config.static_forward_context
        tensors = {}
        specs = []
        for gid, group in enumerate(config.kv_cache_groups):
            page_size = group.kv_cache_spec.page_size_bytes
            storages = {}
            for layer_name in group.layer_names:
                if layer_name not in contexts:
                    raise ValueError(f"No bound KV state for layer {layer_name}")
                state = contexts[layer_name].kv_cache[0]
                first = state[0] if isinstance(state, (list, tuple)) else state
                from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec
                block_axis = None
                factor = 1
                dense = True
                if isinstance(group.kv_cache_spec, MambaSpec):
                    if any(t.stride(0) * t.element_size() != page_size for t in state):
                        raise ValueError("GDN state does not use the expected allocator-page stride")
                elif isinstance(group.kv_cache_spec, AttentionSpec):
                    attn_group = next(g for g in runner._kv_cache_spec_attn_group_iterator()
                                      if layer_name in g.layer_names)
                    kernel_size = runner._init_kernel_block_sizes[gid]
                    factor = group.kv_cache_spec.block_size // kernel_size
                    sentinel = 100003
                    probe = attn_group.backend.get_kv_cache_shape(sentinel, kernel_size,
                        group.kv_cache_spec.num_kv_heads, group.kv_cache_spec.head_size,
                        cache_dtype_str=runner.cache_config.cache_dtype)
                    block_axis = probe.index(sentinel)
                    dense = first.stride(block_axis) * first.element_size() * factor == page_size
                else:
                    raise ValueError("unvalidated KV-cache spec type")
                storage = first.untyped_storage()
                shared_names = next(sorted(t.shared_by) for t in config.kv_cache_tensors
                                    if layer_name in t.shared_by)
                name = "|".join(shared_names)
                if name not in storages:
                    raw = torch.empty(0, dtype=torch.uint8, device=first.device).set_(
                        storage, 0, (storage.nbytes(),), (1,))
                    if raw.numel() % page_size:
                        raise ValueError("allocator storage is not a whole number of pages")
                    storages[name] = (PageStorage(raw, page_size) if dense else
                                      PageStorage(first, page_size, block_axis, factor))
                elif not dense:
                    previous = storages[name].tensor
                    if (first.data_ptr(), first.shape, first.stride()) != (
                            previous.data_ptr(), previous.shape, previous.stride()):
                        raise ValueError("shared planar KV aliases do not use the same logical page view")
            specs.append({"gid": gid, "type": type(group.kv_cache_spec).__name__,
                          "spec": str(group.kv_cache_spec), "page_size": page_size,
                          "storages": sorted(storages),
                          **({"page_views": {name: value.layout for name, value in storages.items()}}
                             if any(value.layout for value in storages.values()) else {})})
            tensors[gid] = storages
        layout = hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest()
        if action == "layout":
            return {"rank": rank, "layout": layout, "specs": specs, "physical_page_layout_validated": True}

        def current_bytes():
            return sum(v["bytes"] for key, v in host.items() if key != entry_id)

        if action == "snapshot":
            nbytes = sum(len(g["blocks"]) * specs[g["gid"]]["page_size"] *
                         len(tensors[g["gid"]]) for g in groups)
            if current_bytes() + nbytes > int(settings["cpu_cache_bytes"]):
                raise MemoryError("bounded CPU cache is full; release a CPU entry first")
            records = []
            torch.cuda.synchronize()
            for group in groups:
                gid = group["gid"]
                page = specs[gid]["page_size"]
                for name, storage in tensors[gid].items():
                    pages = torch.empty((len(group["blocks"]), page), dtype=torch.uint8,
                                        device="cpu", pin_memory=True)
                    for row, block in enumerate(group["blocks"]):
                        pages[row].copy_(storage.read(block["block_id"]), non_blocking=False)
                    records.append({"gid": gid, "name": name, "data": pages,
                                    "digest": digest_tensor(pages)})
            host[entry_id] = {"layout": layout, "specs": specs, "records": records, "bytes": nbytes}
        elif action == "load_artifact":
            path = file_path()
            if path.stat().st_size > int(settings["cpu_cache_bytes"]) + 16 * 1024 * 1024:
                raise ValueError("artifact exceeds CPU-cache limit")
            value = torch.load(path, map_location="cpu", weights_only=True)
            if value["layout"] != layout:
                raise ValueError("incompatible KV/GDN/MTP layout (including K and TP)")
            actual_bytes = sum(rec["data"].numel() * rec["data"].element_size() for rec in value["records"])
            if actual_bytes != value["bytes"]:
                raise ValueError("CPU cache byte accounting mismatch")
            if current_bytes() + actual_bytes > int(settings["cpu_cache_bytes"]):
                raise MemoryError("bounded CPU cache is full")
            expected_records = {(group["gid"], name) for group in groups for name in tensors[group["gid"]]}
            actual_records = [(rec["gid"], rec["name"]) for rec in value["records"]]
            if len(actual_records) != len(expected_records) or set(actual_records) != expected_records:
                raise ValueError("artifact omits/duplicates a KV, GDN or MTP storage")
            for rec in value["records"]:
                if rec["data"].dtype != torch.uint8 or rec["data"].ndim != 2:
                    raise ValueError("invalid byte-page tensor")
                if digest_tensor(rec["data"]) != rec["digest"]:
                    raise ValueError("corrupt cache artifact")
            host[entry_id] = value
        elif action not in {"restore", "save_artifact", "inspect"}:
            raise ValueError(f"unknown worker cache action {action}")

        value = host.get(entry_id)
        if value is None:
            raise KeyError(f"no CPU tensor entry {entry_id}")
        if value["layout"] != layout:
            raise ValueError("CPU cache layout no longer matches the engine")
        if action == "restore":
            mapping = {g["gid"]: g["blocks"] for g in groups}
            # Validate every record before modifying any GPU tensor.
            for rec in value["records"]:
                blocks = mapping[rec["gid"]]
                page = specs[rec["gid"]]["page_size"]
                if rec["data"].shape != (len(blocks), page):
                    raise ValueError("CPU pages do not match the reserved destination slots")
                if digest_tensor(rec["data"]) != rec["digest"]:
                    raise ValueError("CPU tensor checksum mismatch")
            torch.cuda.synchronize()
            for rec in value["records"]:
                storage = tensors[rec["gid"]][rec["name"]]
                page = specs[rec["gid"]]["page_size"]
                for row, block in enumerate(mapping[rec["gid"]]):
                    storage.write(block["block_id"], rec["data"][row])
            torch.cuda.synchronize()
            # D2H readback verifies the installed bytes, not just a successful RPC.
            for rec in value["records"]:
                storage = tensors[rec["gid"]][rec["name"]]
                page = specs[rec["gid"]]["page_size"]
                check = torch.stack([storage.read(b["block_id"]).cpu()
                                     for b in mapping[rec["gid"]]])
                if digest_tensor(check) != rec["digest"]:
                    raise RuntimeError("destination GPU byte verification failed")
        elif action == "save_artifact":
            path = file_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(value, path)
        return {"rank": rank, "layout": layout, "bytes": value["bytes"],
                "records": [{"gid": rec["gid"], "storage": rec["name"],
                             "pages": len(rec["data"]), "sha256": rec["digest"]}
                            for rec in value["records"]],
                "verified_gpu_bytes": action == "restore"}
