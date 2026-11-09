"""Standalone real-weight vLLM V1 instance service.

Run with ``python -m coreflow_mtp.runtime.instance --model /path/to/model ...``.
Only the lifespan of the app imports/starts vLLM; CPU tests inject a backend
directly into create_app rather than exposing a production fake-engine flag.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager, suppress
import json
import logging
from pathlib import Path
import time
import uuid

import httpx

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse

from .backend import GenerationBackend, VllmV1Backend
from .config import InstanceConfig
from .schemas import (GenerateRequest, TokenizeRequest, CacheKeyRequest, CacheMigrationRequest,
                      CacheLookupRequest, ToolObservationRequest)

LOGGER = logging.getLogger(__name__)


class InstanceRuntime:
    def __init__(self, config: InstanceConfig, backend: GenerationBackend) -> None:
        self.config = config
        self.backend = backend
        self.ready = False
        self.active: dict[str, str] = {}
        self.completed_requests = 0
        self.failed_requests = 0
        self.cancelled_requests = 0
        self.input_tokens_total = 0
        self.output_tokens_total = 0
        self.cached_tokens_total = 0
        self.request_seconds_total = 0.

    def capabilities(self) -> dict[str, bool]:
        return {
            "real_generation": True,
            "mtp": self.config.speculative_k > 0,
            "prefix_caching": self.config.enable_prefix_caching,
            "cache_state": self.config.enable_cache_management,
            "retained_gpu_entries": self.config.enable_cache_management,
            "cpu_swap": self.config.enable_cache_management,
            "cache_migration": self.config.enable_cache_management,
            "predictive_cache": self.config.cache_policy_profile is not None,
            "cache_affinity_routing": self.config.cache_affinity_routing,
            "dynamic_speculative_k": False,
        }

    def status(self) -> dict:
        return {
            "status": "ok" if self.ready else "not_ready",
            "ready": self.ready, "instance_id": self.config.instance_id,
            "agent_id": self.config.agent_id,
            "model": self.config.model,
            "max_model_len": self.config.max_model_len,
            "active_requests": len(self.active),
            "completed_requests": self.completed_requests,
            "failed_requests": self.failed_requests,
            "cancelled_requests": self.cancelled_requests,
            "speculative_k": self.config.speculative_k,
            "speculative_tokens": self.config.speculative_k,
            "context_range": self.config.context_range,
            "capabilities": self.capabilities(),
            "configuration": self.config.public_dict(),
            "backend_signature": self.config.backend_signature,
            "backend": self.backend.metadata(),
            "cache_transport": ("same_host_shared_filesystem" if self.config.enable_cache_management else None),
            "cache_scope": "completed_native_prefixes_not_running_request_handoff",
        }

    async def _abort_safely(self, engine_request_id: str) -> None:
        try:
            await asyncio.wait_for(self.backend.abort(engine_request_id), timeout=10)
        except Exception:
            LOGGER.exception("Failed to abort engine request %s", engine_request_id)

    async def generate(self, req: GenerateRequest) -> dict:
        if not self.ready:
            raise HTTPException(503, "engine is not ready")
        if req.agent_id is not None and req.agent_id != self.config.agent_id:
            raise HTTPException(422, "agent_id does not match this instance")
        if req.input_semantics == "synthetic_lengths_ar" and self.config.speculative_k:
            raise HTTPException(422, "synthetic length replay is AR-only and cannot measure MTP acceptance")
        if (req.keep_cache_in_gpu or not req.free_cache) and not self.config.enable_cache_management:
            raise HTTPException(422, "start with --enable-cache-management to request explicit cache ownership")
        if req.cache_policy == "predictive" and not self.config.cache_policy_profile:
            raise HTTPException(422, "predictive policy requires --cache-policy-profile")
        request_id = req.request_id or uuid.uuid4().hex
        if request_id in self.active:
            raise HTTPException(409, "request_id is already active on this instance")
        if len(self.active) >= self.config.max_pending_requests:
            raise HTTPException(429, "instance request queue is full")

        # Use a unique engine ID even if a client reuses a completed public ID.
        # This prevents a delayed abort/output from affecting a later request.
        engine_id = f"{self.config.instance_id}:{uuid.uuid4().hex}"
        self.active[request_id] = engine_id
        started = time.perf_counter()
        successful = False
        engine_started = False
        try:
            prompt_ids = (list(req.prompt_token_ids) if req.prompt_token_ids is not None
                          else await self.backend.tokenize(req.prompt or ""))
            prompt_len = len(prompt_ids)
            if prompt_len < 1:
                raise HTTPException(422, "tokenized prompt is empty")
            if prompt_len < self.config.context_lower or (
                self.config.context_upper is not None and prompt_len >= self.config.context_upper
            ):
                allowed = False
                if self.config.cache_affinity_routing:
                    hit = await self.backend.cache_command({"op": "lookup", "token_ids": prompt_ids})
                    allowed = max(hit.get("gpu_tokens", 0), hit.get("cpu_tokens", 0)) > 0
                if not allowed:
                    raise HTTPException(422, "prompt length is outside this instance's [lower, upper) range without native cache affinity")
            if prompt_len + req.max_tokens > self.config.max_model_len:
                raise HTTPException(422, "prompt_tokens + max_tokens exceeds max_model_len; no truncation is performed")
            engine_started = True
            result = await asyncio.wait_for(
                self.backend.generate(req, prompt_ids, engine_id),
                timeout=self.config.request_timeout_seconds,
            )
            if result.prompt_token_ids != prompt_ids:
                raise RuntimeError("engine returned different prompt token IDs")
            if len(result.token_ids) > req.max_tokens:
                raise RuntimeError("engine exceeded the requested output-token budget")
            successful = True
            self.completed_requests += 1
            self.input_tokens_total += prompt_len
            self.output_tokens_total += len(result.token_ids)
            self.cached_tokens_total += result.num_cached_tokens or 0
            self.request_seconds_total += time.perf_counter() - started
            return {
                "success": True, "request_id": request_id,
                "instance_id": self.config.instance_id,
                "agent_id": self.config.agent_id, "workflow_id": req.workflow_id,
                "query_id": req.query_id, "invocation_id": req.invocation_id,
                "repeat": req.repeat, "text": result.text,
                "prompt_token_ids": result.prompt_token_ids,
                "input_token_ids": result.prompt_token_ids,
                "token_ids": result.token_ids, "output_token_ids": result.token_ids,
                "generated_ids": result.token_ids,
                "prompt_tokens": prompt_len, "num_input_tokens": prompt_len,
                "completion_tokens": len(result.token_ids),
                "num_output_tokens": len(result.token_ids),
                "finish_reason": result.finish_reason, "stop_reason": result.stop_reason,
                "elapsed_seconds": time.perf_counter() - started,
                "num_cached_tokens": result.num_cached_tokens,
                "cache": result.cache_status,
                "cache_evidence_source": (
                    "vllm_request_output" if result.num_cached_tokens is not None else "not_reported"
                ),
                "speculative_k": self.config.speculative_k,
                "speculative_tokens": self.config.speculative_k,
                "backend_signature": self.config.backend_signature,
            }
        except asyncio.TimeoutError as exc:
            self.failed_requests += 1
            raise HTTPException(504, "engine generation exceeded request timeout") from exc
        except asyncio.CancelledError:
            self.cancelled_requests += 1
            raise
        except HTTPException:
            self.failed_requests += 1
            raise
        except ValueError as exc:
            self.failed_requests += 1
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            self.failed_requests += 1
            LOGGER.exception("Generation failed for public request %s", request_id)
            raise HTTPException(500, "engine generation failed; inspect instance logs") from exc
        finally:
            try:
                if engine_started and not successful:
                    # wait_for cancellation and HTTP disconnect both reach here.
                    await asyncio.shield(self._abort_safely(engine_id))
            finally:
                self.active.pop(request_id, None)


async def _run_until_disconnect(runtime: InstanceRuntime, req: GenerateRequest,
                                http_request: Request) -> dict:
    stop_watching = asyncio.Event()

    async def watch_disconnect() -> None:
        while not stop_watching.is_set() and not await http_request.is_disconnected():
            await asyncio.sleep(0.1)

    generation = asyncio.create_task(runtime.generate(req))
    disconnected = asyncio.create_task(watch_disconnect())
    try:
        done, _ = await asyncio.wait({generation, disconnected},
                                     return_when=asyncio.FIRST_COMPLETED)
        if generation in done:
            return await generation
        raise HTTPException(499, "client disconnected; generation aborted")
    finally:
        # Starlette uses AnyIO cancellation scopes in is_disconnected(), which
        # can consume a native task cancellation.  The explicit stop flag
        # prevents the watcher from continuing after a completed generation.
        stop_watching.set()
        for task in (generation, disconnected):
            if not task.done():
                task.cancel()
        await asyncio.gather(generation, disconnected, return_exceptions=True)


def create_app(config: InstanceConfig, *, backend: GenerationBackend | None = None) -> FastAPI:
    runtime = InstanceRuntime(config, backend if backend is not None else VllmV1Backend(config))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            await runtime.backend.start()
            runtime.ready = True
            yield
        finally:
            runtime.ready = False
            for engine_id in list(runtime.active.values()):
                await runtime._abort_safely(engine_id)
            await runtime.backend.shutdown()

    app = FastAPI(title="CoreFlow MTP V1 instance", lifespan=lifespan)
    app.state.runtime = runtime

    @app.get("/health")
    async def health():
        if not runtime.ready:
            raise HTTPException(503, "engine is not ready")
        try:
            await runtime.backend.check_health()
        except Exception as exc:
            runtime.ready = False
            raise HTTPException(503, "engine health check failed") from exc
        return runtime.status()

    @app.get("/status")
    async def status():
        return runtime.status()

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        from .metrics import render_metrics
        cache = {}
        if config.enable_cache_management:
            cache = await runtime.backend.cache_command({"op": "state"})
            cache["gpu_entries"] = [e for e in cache["entries"] if e["location"] == "gpu"]
            cache["cpu_entries"] = [e for e in cache["entries"] if e["location"] == "cpu"]
        return PlainTextResponse(render_metrics(runtime, cache), media_type="text/plain; version=0.0.4")

    @app.get("/capabilities")
    async def capabilities():
        return {"capabilities": runtime.capabilities()}

    @app.get("/registration")
    async def registration():
        return {
            "instance_id": config.instance_id, "agent_id": config.agent_id,
            "host": config.host, "port": config.port,
            "context_range": config.context_range,
            "num_gpus": config.tensor_parallel_size,
            "speculative_k": config.speculative_k,
            "speculative_tokens": config.speculative_k,
            "capabilities": runtime.capabilities(),
            "backend_signature": config.backend_signature,
        }

    @app.get("/cache_state")
    async def cache_state():
        if config.enable_cache_management:
            state = await cache_command({"op": "state"})
            return dict(state, instance_id=config.instance_id, available=True,
                capabilities=runtime.capabilities(), active_requests=len(runtime.active),
                gpu_entries=[entry for entry in state["entries"] if entry["location"] == "gpu"],
                cpu_entries=[entry for entry in state["entries"] if entry["location"] == "cpu"])
        return {
            "instance_id": config.instance_id,
            "active_requests": len(runtime.active),
            "available": False,
            "prefix_caching_enabled": config.enable_prefix_caching,
            "capabilities": runtime.capabilities(),
            "note": "No per-prefix residency directory is exposed. APC is evictable best-effort reuse, not retained workflow cache. Per-request num_cached_tokens is reported only after actual generation.",
        }

    async def cache_command(payload):
        if not config.enable_cache_management:
            raise HTTPException(501, "explicit cache management is disabled")
        if not runtime.ready:
            raise HTTPException(503, "engine is not ready")
        try:
            return await runtime.backend.cache_command(payload)
        except Exception as exc:
            LOGGER.exception("Cache transaction failed: %s", payload.get("op"))
            message = str(exc)
            raise HTTPException(409 if "quiescent" in message else 422, message) from exc

    @app.post("/cache/release")
    async def release_cache(req: CacheKeyRequest):
        return await cache_command(dict(op="release", **req.model_dump()))

    @app.post("/cache/lookup")
    async def lookup_cache(req: CacheLookupRequest):
        return await cache_command(dict(op="lookup", token_ids=req.prompt_token_ids))

    @app.post("/cache/offload")
    async def offload_cache(req: CacheKeyRequest):
        return await cache_command(dict(op="offload", **req.model_dump()))

    @app.post("/cache/restore")
    async def restore_cache(req: CacheKeyRequest):
        return await cache_command(dict(op="restore", **req.model_dump()))

    @app.post("/cache/export")
    async def export_cache(req: CacheKeyRequest):
        return await cache_command(dict(op="export", **req.model_dump()))

    @app.get("/cache/swap_profile")
    async def swap_profile():
        return await cache_command({"op": "swap_profile"})

    @app.post("/cache/tool_observation")
    async def observe_tool(req: ToolObservationRequest):
        profile = getattr(runtime.backend, "cache_policy", None)
        if profile is None:
            raise HTTPException(501, "predictive cache policy is not configured")
        if req.agent_id != config.agent_id:
            raise HTTPException(422, "tool observation agent differs from this instance")
        try:
            return profile.observe(req.agent_id, req.tool_name, req.duration_seconds)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    def discard_artifact(manifest):
        root = Path(config.cache_storage_root).resolve()
        directory = Path(manifest["artifact_dir"]).resolve()
        if (directory.parent != root or len(directory.name) != 32 or
            any(char not in "0123456789abcdef" for char in directory.name)):
            raise ValueError("not a generated cache-transport artifact directory")
        for rank in range(config.tensor_parallel_size):
            path = directory / f"rank_{rank}.pt"
            if path.exists():
                path.unlink()
        directory.rmdir()

    @app.post("/migrate_cache")
    async def migrate_cache(http_request: Request):
        if not config.enable_cache_management:
            raise HTTPException(501, "explicit cache management is disabled")
        from pydantic import ValidationError
        try:
            req = CacheMigrationRequest.model_validate(await http_request.json())
        except (ValidationError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        key = {"query_id": req.query_id, "invocation_id": req.invocation_id}
        manifest = req.manifest
        if req.source_url is not None:
            # First transport is explicitly same-host/shared-filesystem. Do not
            # expose an unauthenticated arbitrary network-fetch endpoint.
            from urllib.parse import urlsplit
            parsed = urlsplit(req.source_url)
            if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
                raise HTTPException(422, "source must be a same-host loopback instance")
            if req.source_url.rstrip("/") == f"http://{config.host}:{config.port}":
                raise HTTPException(422, "source and destination must differ")
            async with httpx.AsyncClient(timeout=180) as client:
                response = await client.post(req.source_url.rstrip("/") + "/cache/export", json=key)
                if response.status_code != 200:
                    raise HTTPException(409, f"source export failed: {response.text}")
                manifest = response.json()
        result = await cache_command(dict(op="import", manifest=manifest,
            keep_cache_in_gpu=req.keep_cache_in_gpu, **key))
        source_released = False
        release_error = None
        if req.move and result.get("imported"):
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    response = await client.post(req.source_url.rstrip("/") + "/cache/release", json=key)
                    response.raise_for_status()
                    source_released = bool(response.json().get("released"))
            except Exception as exc:
                release_error = str(exc)
        artifact_error = None
        if req.source_url is not None and result.get("imported") and "artifact_dir" in manifest:
            try:
                discard_artifact(manifest)
            except (OSError, ValueError) as exc:
                artifact_error = str(exc)
        return dict(success=bool(result.get("imported")), source_released=source_released,
                    source_release_error=release_error, artifact_cleanup_error=artifact_error,
                    device="GPU" if req.keep_cache_in_gpu else "CPU", **result)

    @app.post("/generate")
    async def generate(req: GenerateRequest, request: Request):
        return await _run_until_disconnect(runtime, req, request)

    @app.post("/tokenize")
    async def tokenize(req: TokenizeRequest):
        if not runtime.ready:
            raise HTTPException(503, "engine is not ready")
        token_ids = await runtime.backend.tokenize(req.prompt)
        return {"prompt_token_ids": token_ids, "prompt_tokens": len(token_ids),
                "token_ids": token_ids, "num_tokens": len(token_ids),
                "model": config.model, "backend_signature": config.backend_signature}

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--instance-id", default="instance-0")
    parser.add_argument("--agent-id", default="default")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--context-lower", type=int, default=0)
    parser.add_argument("--context-upper", type=int)
    parser.add_argument("--speculative-k", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--kv-cache-memory-bytes", type=int)
    for option, default in (("enable-chunked-prefill", True), ("enable-prefix-caching", False),
                            ("enforce-eager", True), ("async-scheduling", False),
                            ("language-model-only", True)):
        parser.add_argument(f"--{option}", action=argparse.BooleanOptionalAction, default=default)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--request-timeout-seconds", type=float, default=600)
    parser.add_argument("--max-pending-requests", type=int, default=128)
    parser.add_argument("--enable-cache-management", action="store_true")
    parser.add_argument("--cpu-cache-bytes", type=int, default=2147483648)
    parser.add_argument("--retained-gpu-fraction", type=float, default=.4)
    parser.add_argument("--cache-storage-root", default="/root/nfs/sc_expansion/coreflow_mtp/run/cache_store")
    parser.add_argument("--cache-policy-profile", help="calibrated policy/AR profile bundle for predictive keep-alive")
    parser.add_argument("--cache-affinity-routing", action="store_true")
    parser.add_argument("--execution-trace", help="opt-in synchronized worker trace; instrumentation changes timings")
    parser.add_argument("--print-config", action="store_true", help="validate/print configuration without importing vLLM or starting CUDA")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = vars(parse_args(argv))
    print_config = args.pop("print_config")
    config = InstanceConfig(**args)
    if print_config:
        print(json.dumps({"configuration": config.public_dict(),
                          "engine_kwargs": config.engine_kwargs(),
                          "backend_signature": config.backend_signature}, indent=2, ensure_ascii=False))
        return
    import uvicorn
    # Exactly one web worker owns one engine; multiple uvicorn workers would
    # each load the full model and unexpectedly exhaust GPU memory.
    uvicorn.run(create_app(config), host=config.host, port=config.port, workers=1)


if __name__ == "__main__":
    main()
