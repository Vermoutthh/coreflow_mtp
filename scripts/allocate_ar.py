"""Original AR profiling -> request/invocation capacities -> DP/global allocation.

This is a predicted-capacity path, not a measured MTP table. It requires a
checked new-runtime Llama/K=0 profile and preserves complete provenance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coreflow_mtp.allocator.capacity import CapacityTable, allocate_from_capacity_table
from coreflow_mtp.allocator.request_analyzer import RequestAnalyzer
from coreflow_mtp.allocator.throughput_estimator import ThroughputEstimator
from coreflow_mtp.predictor.ar import ARLatencyPredictor
from coreflow_mtp.runtime.config import InstanceConfig


def build_table(bundle_path, workload, *, kv_bytes, chunk_size, context_step,
                max_context_length, reuse=True, interference=True, retained_gpu_fraction=.8):
    predictor = ARLatencyPredictor(bundle_path)
    geometry = predictor.geometry
    if (not 0 < max_context_length <= min(geometry.max_model_len,
            predictor.metadata["fit_domain"]["max_context"])
            or max_context_length % context_step):
        raise ValueError("context grid must stay within model and calibrated profile limits")
    if not 1 < chunk_size <= predictor.metadata["fit_domain"]["max_query"]:
        raise ValueError("chunk size is outside calibrated query budget")
    if kv_bytes < geometry.kv_bytes_per_token:
        raise ValueError("KV budget cannot hold even one model token")
    analyzer = RequestAnalyzer(str(workload))
    for nodes in analyzer.node_list.values():
        for node in nodes:
            incremental, output, cached = analyzer._token_counts(node)
            if incremental + cached >= max_context_length or incremental + cached + output > max_context_length:
                raise ValueError("workload contains requests outside the planned context/model domain; do not silently drop long requests")
    # Runtime blocks need a small allocator/null-page reserve, not the old
    # model-specific 216K-token constant.
    token_capacity = max(0, kv_bytes // geometry.kv_bytes_per_token - geometry.block_size)
    estimator = ThroughputEstimator(predictor, analyzer, chunk_size=chunk_size,
        context_step=context_step, max_context_length=max_context_length)
    records = []
    for agent in analyzer.agents:
        values = estimator.attain_instance_throughput(geometry.num_layers, token_capacity,
                                                     agent, interference=interference, reuse=reuse)
        records.extend({"agent_id": agent, "context_lower": lo * context_step,
            "context_upper": hi * context_step, "speculative_tokens": 0,
            "normalized_invocations_per_second": capacity} for (lo, hi), capacity in values.items())
    config = InstanceConfig(model=geometry.model, speculative_k=0, dtype=geometry.dtype,
        max_model_len=max_context_length, max_num_seqs=predictor.max_decode_batch,
        max_num_batched_tokens=chunk_size, kv_cache_memory_bytes=kv_bytes,
        enable_prefix_caching=True, enable_cache_management=True, retained_gpu_fraction=retained_gpu_fraction)
    metadata = {"model": geometry.model, "engine_version": "0.18.1",
        "hardware": predictor.metadata["hardware"], "dtype": geometry.dtype,
        "gpus_per_instance": 1, "workload_id": hashlib.sha256(Path(workload).read_bytes()).hexdigest(),
        "provenance": {"profile_bundle": str(Path(bundle_path).resolve()),
            "profile_bundle_sha256": hashlib.sha256(Path(bundle_path).read_bytes()).hexdigest(),
            "workload": str(Path(workload).resolve()), "reuse": reuse, "interference": interference,
            "model": "original_coreflow_AR_hierarchical_approximation",
            "cache_capacity_tokens": token_capacity, "predictions_not_observations": True},
        "capacity_source": "analytical_ar_profile",
        "execution_config": {"chunked_prefill": True, "prefix_caching": True,
            "enforce_eager": True, "cache_management": True,
            "retained_gpu_fraction": retained_gpu_fraction,
            "max_num_seqs": predictor.max_decode_batch, "max_num_batched_tokens": chunk_size,
            "max_model_len": max_context_length, "kv_cache_memory_bytes": kv_bytes,
            "backend_signature": config.backend_signature}}
    table = CapacityTable(metadata, records, context_step, max_context_length)
    return table, analyzer.avg_num_invocations(), estimator.last_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-bundle", required=True)
    parser.add_argument("--workload", required=True, help="original node_list-style agent traces")
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--kv-cache-memory-bytes", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--context-step", type=int, default=2000)
    parser.add_argument("--max-context-length", type=int, default=128000)
    parser.add_argument("--max-instance-group", type=int, default=3)
    parser.add_argument("--reuse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retained-gpu-fraction", type=float, default=.8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        parser.error("refusing to overwrite an existing allocation")
    table, invocations, diagnostics = build_table(args.profile_bundle, args.workload,
        kv_bytes=args.kv_cache_memory_bytes, chunk_size=args.chunk_size,
        context_step=args.context_step, max_context_length=args.max_context_length,
        reuse=args.reuse, interference=args.interference, retained_gpu_fraction=args.retained_gpu_fraction)
    result = allocate_from_capacity_table(table, args.gpus, invocations,
        max_instance_group=args.max_instance_group, allowed_k={0})
    result["estimator_diagnostics_last_range"] = diagnostics
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Wrote {output}; source=analytical_ar_profile; predicted workflow/s={result['predicted_workflow_throughput']:.6g}")


if __name__ == "__main__":
    main()
