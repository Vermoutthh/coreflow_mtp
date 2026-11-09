"""Original CoreFlow component profiles, ported to the pinned V1 kernel API."""
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from coreflow_mtp.predictor.linear import LinearRegression
from coreflow_mtp.profiler.config import ARModelGeometry, profiling_grid
from coreflow_mtp.profiler.kernels import ARKernelProfiler


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def fit_swap(rows, key):
    model = LinearRegression().fit([[r["bytes"]] for r in rows], [r[key]["seconds"] for r in rows])
    return {"intercept_seconds": max(0., model.intercept_),
            "seconds_per_byte": max(0., float(model.coef_[0]))}


def run_profile(model, output, *, smoke=False, device="cuda:0", warmup=5, repeat=10,
                blocks=100000, timing_protocol="coreflow_forward", grid=None):
    geometry = ARModelGeometry.load(model)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    grid = grid or profiling_grid(smoke)
    write_json(out / "geometry.json", geometry.public_dict())
    write_json(out / "grid.json", grid)
    runner = ARKernelProfiler(geometry, device, warmup, repeat, blocks, timing_protocol)
    raw = {"decode": {}, "prefill": {}, "baseline": {}, "mixed": {}}
    attention = {"decode": {}, "prefill": {}}
    print("[profile] AR decode attention", flush=True)
    for b in grid["batches"]:
        raw["decode"][str(b)], attention["decode"][str(b)] = {}, {}
        for c in grid["contexts"]:
            observed = runner.measure(runner.attention_call([c] * b))
            raw["decode"][str(b)][str(c)] = observed
            attention["decode"][str(b)][str(b * c)] = observed["seconds"]
    print("[profile] AR prefill attention", flush=True)
    for q in grid["queries"]:
        raw["prefill"][str(q)], attention["prefill"][str(q)] = {}, {}
        for c in grid["contexts"]:
            if q > c:
                continue
            observed = runner.measure(runner.attention_call([c], [q]))
            raw["prefill"][str(q)][str(c)] = observed
            attention["prefill"][str(q)][str(c)] = observed["seconds"]
    write_json(out / "attn_profile.json", attention)
    write_json(out / "raw_samples.json", raw)
    print("[profile] per-long-context homogeneous B scan for intercept", flush=True)
    baseline = {}
    for length in grid["long_contexts"]:
        observations = [runner.measure(runner.attention_call([length] * b)) for b in grid["baseline_batches"]]
        fitted = LinearRegression().fit([[b] for b in grid["baseline_batches"]], [r["seconds"] for r in observations])
        baseline[str(length)] = {"initial": fitted.intercept_, "diff": float(fitted.coef_[0])}
        raw["baseline"][str(length)] = {str(b): value for b, value in zip(grid["baseline_batches"], observations)}
    write_json(out / "baseline_profile.json", baseline)
    print("[profile] mixed - short_group - long + per-L intercept", flush=True)
    overhead, components = {}, {}
    for b in grid["mixed_batches"]:
        overhead[str(b)], components[str(b)] = {}, {}
        # A single measured short-group value per (B,S), as in CoreFlow reuse.
        shorts = {s: runner.measure(runner.attention_call([s] * (b - 1))) for s in grid["short_contexts"]}
        for length in grid["long_contexts"]:
            long = runner.measure(runner.attention_call([length]))
            overhead[str(b)][str(length)], components[str(b)][str(length)] = {}, {}
            for short in grid["short_contexts"]:
                if short >= length:
                    continue
                mix = runner.measure(runner.attention_call([length] + [short] * (b - 1)))
                signed = mix["seconds"] - shorts[short]["seconds"] - long["seconds"] + baseline[str(length)]["initial"]
                overhead[str(b)][str(length)][str(short)] = max(0., signed)
                components[str(b)][str(length)][str(short)] = {"mixed": mix, "long": long,
                    "short": shorts[short], "intercept_seconds": baseline[str(length)]["initial"],
                    "signed_overhead_seconds": signed, "paper_overhead_seconds": max(0., signed)}
    raw["mixed"] = components
    write_json(out / "interference_profile.json", overhead)
    write_json(out / "raw_samples.json", raw)
    print("[profile] Llama layer excluding attention", flush=True)
    non_attention = runner.non_attention(grid["non_attention_batches"])
    write_json(out / "mlp_profile.json", {b: row["seconds"] for b, row in non_attention.items()})
    write_json(out / "non_attention_samples.json", non_attention)
    print("[profile] measured model-sized KV D2H/H2D copies", flush=True)
    swaps = runner.swap(grid["swap_tokens"])
    write_json(out / "swap_profile.json", {"scope": "synchronized_raw_tensor_copy_not_RPC_or_hashing",
        "observations": swaps, "d2h": fit_swap(swaps, "d2h"), "h2d": fit_swap(swaps, "h2d")})
    prop = runner.torch.cuda.get_device_properties(device)
    metadata = {"schema_version": 1, "kind": "coreflow_ar_components", "speculative_k": 0,
        "engine_version": importlib.metadata.version("vllm"), "geometry": geometry.public_dict(),
        "hardware": prop.name, "compute_capability": [prop.major, prop.minor],
        "multiprocessors": prop.multi_processor_count, "tensor_parallel_size": 1,
        "attention_backend": "FLASH_ATTN_FA2", "num_splits": 0,
        "timing_protocol": timing_protocol, "dtype": geometry.dtype,
        "warmup": warmup, "repeat": repeat, "pool_blocks": blocks,
        "created_at": datetime.now().astimezone().isoformat(), "smoke_grid": smoke,
        "profiled_decode_batches": grid["batches"],
        "fit_domain": {"max_decode_batch": max(grid["batches"]), "max_query": max(grid["queries"]),
                       "max_non_attention_tokens": max(grid["non_attention_batches"]),
                       "max_context": max(grid["contexts"] + grid["long_contexts"])},
        "files": {name: {"path": name, "sha256": hashlib.sha256((out / name).read_bytes()).hexdigest()}
                  for name in ("attn_profile.json", "interference_profile.json", "mlp_profile.json", "swap_profile.json")},
        "scope": "component_microbench_predictions_require_full_engine_validation"}
    write_json(out / "profile_bundle.json", metadata)
    print(f"[profile] completed: {out / 'profile_bundle.json'}", flush=True)
    return metadata
