"""Allocation-driven process launcher, without legacy import-path injection.

The allocation declares a fixed K per instance group. GPU IDs refer to physical
devices, not an inherited logical device remapping. No engine starts on dry-run.
"""
from __future__ import annotations

import argparse
from contextlib import suppress
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]


def request_json(url, payload=None, timeout=10):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def clean_engine_env(devices):
    env = os.environ.copy()
    # New vLLM must not inherit the old repository or a V0 selection override.
    for key in ("PYTHONPATH", "PYTHONHOME", "VLLM_USE_V1", "VLLM_USE_V2_MODEL_RUNNER"):
        env.pop(key, None)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def parse_devices(value):
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values or any(not item.isdecimal() for item in values) or len(set(values)) != len(values):
        raise ValueError("devices must be distinct physical numeric GPU IDs, e.g. 0,2")
    return values


def plan_instances(data, devices, *, model=None, base_port=18100, max_model_len=8192):
    allocation = data.get("allocation")
    if not isinstance(allocation, dict) or not allocation:
        raise ValueError("allocation must be a nonempty agent -> instances mapping")
    defaults = data.get("config", {})
    pool = list(devices)
    plans = []
    for agent_id, agent in allocation.items():
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("agent IDs must be nonempty strings")
        for group_index, group in enumerate(agent.get("instances", [])):
            tp = int(group.get("gpus_per_instance", defaults.get("gpus_per_instance", 1)))
            count = int(group.get("num_instances", 1))
            k = group.get("speculative_tokens", group.get("speculative_k", defaults.get("speculative_tokens")))
            if k is None:
                raise ValueError("explicit speculative_tokens is required; old allocations are not assumed MTP-ready")
            if isinstance(k, bool) or int(k) != k or int(k) < 0:
                raise ValueError("speculative_tokens must be a nonnegative integer")
            lower, upper = int(group["context_lower"]), int(group["context_upper"])
            limit = int(group.get("max_model_len", max_model_len))
            group_model = group.get("model", model or defaults.get("model"))
            if not group_model:
                raise ValueError("supply model in allocation or --model")
            if tp < 1 or count < 1 or not 0 <= lower < upper <= limit:
                raise ValueError("invalid GPU count, replica count or context range/model limit")
            if "gpus" in group and int(group["gpus"]) != tp * count:
                raise ValueError("group gpus does not equal num_instances * gpus_per_instance")
            for replica in range(count):
                if len(pool) < tp:
                    raise ValueError("not enough devices for the entire allocation; no engine was started")
                port = base_port + len(plans)
                if not 1 <= port <= 65535:
                    raise ValueError("port out of range")
                plans.append({
                    "instance_id": f"agent{len(plans)}-g{group_index}-r{replica}",
                    "agent_id": agent_id, "gpu_devices": pool[:tp], "port": port,
                    "context_lower": lower, "context_upper": upper,
                    "tensor_parallel_size": tp, "speculative_tokens": int(k),
                    "model": str(group_model), "max_model_len": limit,
                })
                pool = pool[tp:]
    if not plans:
        raise ValueError("allocation contains no instance groups")
    return plans


def build_command(plan, args):
    cmd = [args.python, "-I", "-B", str(ROOT / "scripts/serve_instance.py"),
           "--host", "127.0.0.1", "--port", str(plan["port"]),
           "--model", plan["model"], "--instance-id", plan["instance_id"],
           "--agent-id", plan["agent_id"],
           "--context-lower", str(plan["context_lower"]),
           "--context-upper", str(plan["context_upper"]),
           "--speculative-k", str(plan["speculative_tokens"]),
           "--tensor-parallel-size", str(plan["tensor_parallel_size"]),
           "--max-model-len", str(plan["max_model_len"]),
           "--max-num-seqs", str(args.max_num_seqs),
           "--max-num-batched-tokens", str(args.max_num_batched_tokens),
           "--gpu-memory-utilization", str(args.gpu_memory_utilization)]
    if args.kv_cache_memory_bytes is not None:
        cmd += ["--kv-cache-memory-bytes", str(args.kv_cache_memory_bytes)]
    if args.enforce_eager:
        cmd += ["--enforce-eager"]
    else:
        cmd += ["--no-enforce-eager"]
    if args.enable_prefix_caching:
        cmd += ["--enable-prefix-caching"]
    if getattr(args, "enable_cache_management", False):
        cmd += ["--enable-cache-management", "--cpu-cache-bytes", str(args.cpu_cache_bytes),
                "--cache-storage-root", args.cache_storage_root,
                "--retained-gpu-fraction", str(args.retained_gpu_fraction)]
        if getattr(args, "cache_policy_profile", None):
            cmd += ["--cache-policy-profile", args.cache_policy_profile]
        if getattr(args, "cache_affinity_routing", False):
            cmd += ["--cache-affinity-routing"]
        if getattr(args, "execution_trace_dir", None):
            cmd += ["--execution-trace", str(Path(args.execution_trace_dir).resolve() / (plan["instance_id"] + ".jsonl"))]
    if not args.enable_chunked_prefill:
        cmd += ["--no-enable-chunked-prefill"]
    return cmd


def validate_profile_configuration(data, args):
    """Reject known deployment/profile mismatches; do not invent calibration."""
    metadata = data.get("profile_metadata")
    if metadata is None:
        return  # explicit manual plans are permitted, but have no prediction
    if metadata.get("engine_version") != "0.18.1":
        raise ValueError("capacity table engine version differs from the supported vLLM 0.18.1")
    expected_model = str(metadata["model"])
    if args.model is not None and args.model != expected_model:
        raise ValueError("--model differs from the profiled model")
    execution = metadata.get("execution_config", {})
    actual = {
        "chunked_prefill": args.enable_chunked_prefill,
        "prefix_caching": args.enable_prefix_caching,
        "enforce_eager": args.enforce_eager,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_model_len": getattr(args, "max_model_len", None),
        "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
        "cache_management": getattr(args, "enable_cache_management", False),
        "cpu_cache_bytes": getattr(args, "cpu_cache_bytes", 2147483648),
        "retained_gpu_fraction": getattr(args, "retained_gpu_fraction", .4),
    }
    for key, value in actual.items():
        if key in execution and execution[key] != value:
            raise ValueError(f"deployment {key}={value} differs from profiled {execution[key]}")


def _start_ticks(pid):
    try:
        # /proc/<pid>/stat comm can contain spaces and parentheses.
        tail = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return tail[19]
    except (OSError, IndexError):
        return None


def _same_process(item):
    return item.get("start_ticks") is not None and _start_ticks(int(item["pid"])) == item["start_ticks"]


def write_state(path, items):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(items, indent=2, ensure_ascii=False))
    temporary.replace(path)


def _start(args):
    if args.enable_cache_management and not args.enable_prefix_caching:
        raise ValueError("--enable-cache-management requires --enable-prefix-caching")
    devices = parse_devices(args.devices or os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    data = json.loads(Path(args.allocation).read_text())
    validate_profile_configuration(data, args)
    plans = plan_instances(data, devices,
                           model=args.model, base_port=args.base_port,
                           max_model_len=args.max_model_len)
    for plan in plans:
        plan["command"] = build_command(plan, args)
    if args.dry_run:
        print(json.dumps(plans, indent=2, ensure_ascii=False))
        return
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "instances.json"
    if state_path.exists():
        raise ValueError(f"state already exists: {state_path}; stop/use a new output directory")
    # Fail before allocating GPU memory if the controller is absent.
    if args.controller_url:
        request_json(args.controller_url.rstrip("/") + "/health")
    items = []
    try:
        for plan in plans:
            with (out / f"{plan['instance_id']}.log").open("w") as log:
                proc = subprocess.Popen(plan["command"], cwd=str(ROOT),
                                        env=clean_engine_env(plan["gpu_devices"]),
                                        stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
            item = dict(plan, pid=proc.pid, start_ticks=_start_ticks(proc.pid),
                        endpoint=f"http://127.0.0.1:{plan['port']}")
            items.append(item)
            write_state(state_path, items)
            print(f"Starting {plan['instance_id']}: GPU {plan['gpu_devices']}, K={plan['speculative_tokens']}", flush=True)
            deadline = time.monotonic() + args.health_timeout
            health = None
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"{plan['instance_id']} exited {proc.returncode}; see {out}")
                try:
                    health = request_json(item["endpoint"] + "/health", timeout=2)
                    if health.get("ready") is True or health.get("status") == "ok":
                        break
                except (OSError, ValueError, urllib.error.URLError):
                    pass
                time.sleep(0.5)
            else:
                raise TimeoutError(f"engine startup timed out; see {out}")
            if args.controller_url:
                cfg = health.get("configuration", {})
                registration = {
                    "instance_id": plan["instance_id"], "agent_id": plan["agent_id"],
                    "host": "127.0.0.1", "port": plan["port"],
                    "context_range": [plan["context_lower"], plan["context_upper"]],
                    "num_gpus": plan["tensor_parallel_size"],
                    "speculative_tokens": plan["speculative_tokens"],
                    "capabilities": health.get("capabilities", {}),
                    "model": plan["model"], "max_model_len": plan["max_model_len"],
                    "backend_signature": health.get("backend_signature", cfg.get("backend_signature")),
                }
                request_json(args.controller_url.rstrip("/") + "/register", registration)
            item["ready"] = True
            write_state(state_path, items)
        print(f"Ready: {len(items)} instances; state {state_path}", flush=True)
    except BaseException:
        # Only terminate the exact process groups created by this invocation.
        for item in items:
            if _same_process(item):
                with suppress(ProcessLookupError):
                    os.killpg(item["pid"], signal.SIGTERM)
        raise


def _stop(args):
    path = Path(args.state_file).resolve()
    items = json.loads(path.read_text())
    for item in items:
        if _same_process(item):
            with suppress(ProcessLookupError):
                os.killpg(int(item["pid"]), signal.SIGTERM)
            print(f"Stopped owned instance {item['instance_id']}")
        else:
            print(f"Skipped exited/reused PID {item['pid']}")
    # Keep state/logs for audit; future runs use a fresh directory.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start")
    start.add_argument("--allocation", required=True)
    start.add_argument("--model")
    start.add_argument("--devices", help="physical numeric GPU IDs; otherwise CUDA_VISIBLE_DEVICES")
    start.add_argument("--python", default=sys.executable)
    start.add_argument("--controller-url", default="http://127.0.0.1:18500")
    start.add_argument("--base-port", type=int, default=18100)
    start.add_argument("--max-model-len", type=int, default=8192)
    start.add_argument("--max-num-seqs", type=int, default=16)
    start.add_argument("--max-num-batched-tokens", type=int, default=8192)
    start.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    start.add_argument("--kv-cache-memory-bytes", type=int)
    start.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    start.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=False)
    start.add_argument("--enable-cache-management", action="store_true")
    start.add_argument("--cpu-cache-bytes", type=int, default=2147483648)
    start.add_argument("--retained-gpu-fraction", type=float, default=.4)
    start.add_argument("--cache-storage-root", default=str(ROOT / "run/cache_store"))
    start.add_argument("--cache-policy-profile")
    start.add_argument("--cache-affinity-routing", action="store_true")
    start.add_argument("--execution-trace-dir")
    start.add_argument("--enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=True)
    start.add_argument("--health-timeout", type=float, default=600)
    start.add_argument("--out-dir", required=True)
    start.add_argument("--dry-run", action="store_true")
    start.set_defaults(func=_start)
    stop = sub.add_parser("stop")
    stop.add_argument("--state-file", required=True)
    stop.set_defaults(func=_stop)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, RuntimeError, TimeoutError, OSError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
