"""Narrow engine boundary: production always uses real vLLM V1 execution."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .config import InstanceConfig

if TYPE_CHECKING:
    from .schemas import GenerateRequest


@dataclass
class GenerationResult:
    prompt_token_ids: list[int]
    token_ids: list[int]
    text: str
    finish_reason: str | None
    stop_reason: str | int | None = None
    num_cached_tokens: int | None = None
    cache_status: dict | None = None


class GenerationBackend(Protocol):
    """Tests inject an in-process implementation, never via a production flag."""

    async def start(self) -> None: ...
    async def shutdown(self) -> None: ...
    async def check_health(self) -> None: ...
    async def tokenize(self, text: str) -> list[int]: ...
    async def generate(self, request: GenerateRequest, prompt_token_ids: list[int],
                       engine_request_id: str) -> GenerationResult: ...
    async def abort(self, engine_request_id: str) -> None: ...
    def metadata(self) -> dict: ...


class VllmV1Backend:
    def __init__(self, config: InstanceConfig) -> None:
        self.config = config
        self.engine = None
        self._version: str | None = None
        self._module_path: str | None = None
        self._cache_sessions = {}
        self._cache_layout = None
        self.cache_policy = None
        self._maintenance_task = None
        self._maintenance_error = None

    async def start(self) -> None:
        import vllm
        self._version = vllm.__version__
        self._module_path = str(Path(vllm.__file__).resolve())
        if self._version != self.config.expected_vllm_version:
            raise RuntimeError(
                f"Expected vLLM {self.config.expected_vllm_version}, got {self._version} "
                f"from {self._module_path}; use the coreflow_mtp environment."
            )
        if "/CoreFlow_/src/" in self._module_path or "/src/vllm/vllm/" in self._module_path:
            raise RuntimeError(f"Refusing vendored legacy vLLM: {self._module_path}")
        if self.config.enable_cache_management and not Path(self.config.model).is_dir():
            raise ValueError("cache transfer requires a pinned local model snapshot, not a mutable remote model ID")
        if self.config.cache_policy_profile:
            from coreflow_mtp.cache.policy import CachePolicyProfile
            self.cache_policy = CachePolicyProfile.load(self.config.cache_policy_profile,
                self.config.model, self.config.speculative_k, self.config.tensor_parallel_size)
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        # Create inside the serving event loop, as required by AsyncLLM V1.
        self.engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**self.config.engine_kwargs()))
        await self.check_health()
        if self.config.enable_cache_management:
            self._cache_layout = await self.cache_command({"op": "validate"})
            if self.cache_policy is not None:
                self._maintenance_task = asyncio.create_task(self._maintain_keepalive())

    async def shutdown(self) -> None:
        if self._maintenance_task is not None:
            self._maintenance_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._maintenance_task
            self._maintenance_task = None
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None

    async def check_health(self) -> None:
        if self.engine is None:
            raise RuntimeError("engine is not initialized")
        await self.engine.check_health()

    async def tokenize(self, text: str) -> list[int]:
        if self.engine is None:
            raise RuntimeError("engine is not initialized")
        # The resulting IDs, not the source text, are passed to generate: the
        # scheduler and engine therefore see exactly the length we validate.
        return list(await asyncio.to_thread(self.engine.get_tokenizer().encode, text))

    @asynccontextmanager
    async def _cache_session(self, request):
        if not self.config.enable_cache_management or request.query_id is None or request.invocation_id is None:
            yield
            return
        key = str(request.query_id), str(request.invocation_id)
        record = self._cache_sessions.setdefault(key, {"lock": asyncio.Lock(), "users": 0})
        record["users"] += 1
        try:
            async with record["lock"]:
                yield
        finally:
            record["users"] -= 1
            if not record["users"]:
                self._cache_sessions.pop(key, None)

    async def generate(self, request: GenerateRequest, prompt_token_ids: list[int],
                       engine_request_id: str) -> GenerationResult:
        async with self._cache_session(request):
            return await self._generate_native(request, prompt_token_ids, engine_request_id)

    async def _generate_native(self, request: GenerateRequest, prompt_token_ids: list[int],
                               engine_request_id: str) -> GenerationResult:
        if self.engine is None:
            raise RuntimeError("engine is not initialized")
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        adaptive = None
        if request.cache_policy == "predictive" and not request.free_cache:
            if self.cache_policy is None:
                raise ValueError("predictive cache policy needs a calibrated cache-policy profile")
            adaptive = self.cache_policy.request_policy(self.config.agent_id, request.tool_name,
                                                        request.expected_tool_latency_seconds)
            observed_cost = await self.cache_command({"op": "swap_profile"})
            if observed_cost.get("available"):
                adaptive["fallbacks"] = {"d2h": adaptive["d2h"], "h2d": adaptive["h2d"]}
                adaptive.update(d2h=observed_cost["d2h"], h2d=observed_cost["h2d"],
                                provenance=observed_cost["provenance"])

        sampling = SamplingParams(
            n=1, temperature=request.temperature, top_p=request.top_p,
            top_k=request.top_k, max_tokens=request.max_tokens,
            min_tokens=request.min_tokens, stop=request.stop,
            ignore_eos=request.ignore_eos, seed=request.seed,
            output_kind=RequestOutputKind.FINAL_ONLY,
            extra_args={"coreflow_cache": {
                "engine_request_id": engine_request_id,
                "query_id": request.query_id, "invocation_id": request.invocation_id,
                "keep_cache_in_gpu": request.keep_cache_in_gpu, "free_cache": request.free_cache,
                **({"adaptive_policy": adaptive} if adaptive is not None else {}),
            }} if self.config.enable_cache_management and request.query_id is not None and request.invocation_id is not None else None,
        )
        if self.config.enable_cache_management and request.query_id is not None and request.invocation_id is not None:
            await self.cache_when_idle({"op": "prepare", "query_id": request.query_id,
                "invocation_id": request.invocation_id, "token_ids": prompt_token_ids})
        final = None
        async for output in self.engine.generate(
            {"prompt_token_ids": prompt_token_ids}, sampling, engine_request_id,
        ):
            final = output
        if final is None or not final.finished or len(final.outputs) != 1:
            raise RuntimeError("engine did not return exactly one finished completion")
        cache_status = None
        if self.config.enable_cache_management:
            cache_status = await self.cache_command({"op": "report", "engine_request_id": engine_request_id})
            if (not request.free_cache and cache_status.get("stored")
                    and not cache_status.get("keep_cache_in_gpu", request.keep_cache_in_gpu)):
                await self.cache_when_idle({"op": "offload", "query_id": request.query_id,
                                           "invocation_id": request.invocation_id})
                cache_status["location"] = "cpu"
        completion = final.outputs[0]
        return GenerationResult(
            prompt_token_ids=list(final.prompt_token_ids or prompt_token_ids),
            token_ids=list(completion.token_ids), text=completion.text,
            finish_reason=completion.finish_reason, stop_reason=completion.stop_reason,
            num_cached_tokens=getattr(final, "num_cached_tokens", None),
            cache_status=cache_status,
        )

    async def _maintain_keepalive(self):
        while True:
            await asyncio.sleep(.1)
            try:
                await self.cache_command({"op": "expire"})
                self._maintenance_error = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Expose cache maintenance failure without claiming a swap
                # happened or silently destroying a still-valid GPU owner.
                self._maintenance_error = str(exc)
                await asyncio.sleep(.5)

    async def abort(self, engine_request_id: str) -> None:
        if self.engine is not None:
            await self.engine.abort(engine_request_id)

    async def cache_command(self, payload: dict) -> dict:
        if not self.config.enable_cache_management or self.engine is None:
            raise ValueError("explicit cache management is disabled")
        return await self.engine.engine_core.call_utility_async("coreflow_cache", payload)

    async def cache_when_idle(self, payload: dict) -> dict:
        payload = dict(payload, wait_for_idle=True)
        deadline = asyncio.get_running_loop().time() + self.config.request_timeout_seconds
        while True:
            try:
                result = await self.cache_command(payload)
                if not result.get("deferred"):
                    return result
            except Exception as exc:
                if "quiescent" not in str(exc):
                    raise
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("cache transaction timed out waiting for quiescence")
            await asyncio.sleep(.05)

    def metadata(self) -> dict:
        return {"name": "vllm_v1_async", "vllm_version": self._version,
                "vllm_module": self._module_path,
                "cache_layout_validation": self._cache_layout,
                "cache_maintenance_error": self._maintenance_error,
                "backend_signature": self.config.backend_signature}
