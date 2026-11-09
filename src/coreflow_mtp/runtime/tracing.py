"""Opt-in, version-pinned actual worker execution trace.

Synchronization is deliberate diagnostics overhead. This is not claimed to be
unmodified serving latency, executor RPC latency, or draft/verification timing.
"""
import json
from pathlib import Path
import time


def install_worker_trace(worker, trace_path, rank):
    runner = worker.model_runner
    if getattr(runner, "_coreflow_trace_installed", False):
        return
    path = Path(trace_path).resolve()
    path = path.with_name(f"{path.stem}.rank{rank}{path.suffix or '.jsonl'}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x"):
        pass
    original = runner.execute_model

    def execute(*args, **kwargs):
        import torch
        scheduled = args[0] if args else kwargs["scheduler_output"]
        assignments = dict(scheduled.num_scheduled_tokens)
        if not assignments:
            # Native completion/cleanup calls are real, but are not GPU
            # attention/decode work and must not become fake B=0 samples.
            return original(*args, **kwargs)
        new = {r.req_id: r for r in scheduled.scheduled_new_reqs}
        contexts, prefill = [], []
        for request_id in assignments:
            state = runner.requests.get(request_id) or new.get(request_id)
            if state is None:
                contexts.append(None)
                prefill.append(None)
            else:
                prompt = len(state.prompt_token_ids or [])
                contexts.append(prompt + len(getattr(state, "output_token_ids", [])))
                prefill.append(state.num_computed_tokens < prompt)
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = original(*args, **kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        row = {"rank": rank, "request_ids": list(assignments), "batch_size": len(assignments),
               "context_lengths": contexts, "scheduled_tokens": assignments,
               "all_decode": all(value is False for value in prefill),
               "worker_execute_wall_seconds": elapsed,
               "timing_boundary": "synchronized_GPUModelRunner_execute_model_not_executor_RPC",
               "instrumented": True}
        with path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        return result
    runner.execute_model = execute
    runner._coreflow_trace_installed = True
