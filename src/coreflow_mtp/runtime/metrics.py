"""Prometheus text from actual adapter counters and V1 scheduler observations."""
import json


def render_metrics(runtime, cache=None):
    cache = cache or {}
    labels = ",".join(f"{key}={json.dumps(str(value))}" for key, value in (
        ("instance_id", runtime.config.instance_id), ("agent_id", runtime.config.agent_id),
        ("speculative_k", runtime.config.speculative_k)))
    lines = []
    def sample(name, value, kind="gauge"):
        lines.extend([f"# TYPE {name} {kind}", f"{name}{{{labels}}} {value}"])
    sample("coreflow_active_requests", len(runtime.active))
    for name, value in (("completed", runtime.completed_requests), ("failed", runtime.failed_requests),
                        ("cancelled", runtime.cancelled_requests)):
        sample(f"coreflow_requests_{name}_total", value, "counter")
    for name, value in (("input", runtime.input_tokens_total), ("output", runtime.output_tokens_total),
                        ("cached_prefix", runtime.cached_tokens_total)):
        sample(f"coreflow_{name}_tokens_total", value, "counter")
    sample("coreflow_request_latency_seconds_sum", runtime.request_seconds_total, "counter")
    sample("coreflow_gpu_prefix_token_hit_rate", runtime.cached_tokens_total / max(1, runtime.input_tokens_total))
    # This is explicitly token-based, not an undocumented alias for old V0's
    # block-hit gauge (hybrid/speculative blocks can have different sizes).
    sample("coreflow_retained_gpu_entries", len(cache.get("gpu_entries", [])))
    sample("coreflow_retained_cpu_entries", len(cache.get("cpu_entries", [])))
    sample("coreflow_cpu_cache_bytes", sum(rank["bytes"] for rank in cache.get("cpu_cache_workers", [])))
    for key, value in cache.get("counters", {}).items():
        sample(f"coreflow_cache_{key}_total", value, "counter")
    step = cache.get("step_stats")
    if step:
        name = "coreflow_scheduled_requests_per_step"
        lines.append(f"# TYPE {name} histogram")
        for bucket, count in step["buckets"].items():
            lines.append(f'{name}_bucket{{{labels},le="{bucket}"}} {count}')
        lines.append(f'{name}_bucket{{{labels},le="+Inf"}} {step["count"]}')
        lines.append(f'{name}_count{{{labels}}} {step["count"]}')
        lines.append(f'{name}_sum{{{labels}}} {step["sum"]}')
    return "\n".join(lines) + "\n"
