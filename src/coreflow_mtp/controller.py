"""Async control plane for unmodified vLLM V1 adapters.

The adapter returns complete generation results synchronously. This controller,
not a vendored vLLM worker callback, advances the validated workflow DAG.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import json
import time

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from coreflow_mtp.dag.dag import NodeStatus, node_key, token_ids
from coreflow_mtp.dag.query import Query
from coreflow_mtp.instance_manager import InstanceManager, normalize_capabilities
from coreflow_mtp.scheduler import AsyncAgentScheduler
from coreflow_mtp.runtime.schemas import ToolObservationRequest


@dataclass
class QueryState:
    query_id: int
    query: Query
    completed: bool = False
    error: str | None = None
    input_semantics: str = "real_tokens"
    submit_time: float = field(default_factory=time.time)


class AsyncController:
    def __init__(self, http_client=None, routing_policy="strict"):
        self.routing_policy = routing_policy
        self.instance_manager = InstanceManager()
        self._http = http_client
        self._owns_http = http_client is None
        self._schedulers = {}
        self._queries: dict[int, QueryState] = {}
        self._tasks: set[asyncio.Task] = set()

    @property
    def http_client(self):
        if self._http is None:
            # Keep the original shared HTTPX client and default pool sizes.
            # Retire idle sockets before Uvicorn's default 5-second timeout.
            self._http = httpx.AsyncClient(timeout=600.0, limits=httpx.Limits(
                max_connections=100, max_keepalive_connections=20, keepalive_expiry=1.0))
        return self._http

    async def get_scheduler(self, agent_id):
        if agent_id not in self._schedulers:
            self._schedulers[agent_id] = AsyncAgentScheduler(
                agent_id, self.instance_manager, self.http_client, self.routing_policy)
        return self._schedulers[agent_id]

    async def add_query(self, state):
        if state.query_id in self._queries:
            raise ValueError("query_id already exists")
        self._queries[state.query_id] = state

    async def get_query(self, query_id):
        return self._queries.get(query_id)

    async def register_instance(self, payload):
        instance_id = payload["instance_id"]
        previous = self.instance_manager.get(instance_id)
        requested_k = int(payload.get("speculative_tokens", payload.get("speculative_k", 0)))
        if previous is not None and previous.speculative_tokens != requested_k:
            raise ValueError("Registered K differs from the allocation stub")
        endpoint = f"http://{payload.get('host', '127.0.0.1')}:{int(payload['port'])}"
        health, caps, status = await asyncio.gather(*(
            self.http_client.get(endpoint + path)
            for path in ("/health", "/capabilities", "/status")))
        for response in (health, caps, status):
            response.raise_for_status()
        health_data, cap_data, status_data = health.json(), caps.json(), status.json()
        if health_data.get("ready") is False:
            raise ValueError("Instance is not ready")
        if status_data.get("ready") is False or status_data.get("status") in {"starting", "error", "unavailable", "not_ready"}:
            raise ValueError("Instance is not ready")
        actual_k = status_data.get("speculative_tokens", status_data.get("speculative_k"))
        if actual_k is None:
            actual_k = health_data.get("speculative_tokens", health_data.get("speculative_k"))
        if actual_k is None:
            actual_k = cap_data.get("speculative_tokens", cap_data.get("speculative_k"))
        if actual_k is None:
            raise ValueError("Backend must advertise its fixed speculative_tokens")
        if int(actual_k) != requested_k:
            raise ValueError("Requested K does not match the running engine")
        backend_agent = status_data.get("agent_id", cap_data.get("agent_id"))
        if backend_agent is not None and backend_agent != payload["agent_id"]:
            raise ValueError("Registered agent differs from the running instance")
        if status_data.get("instance_id", instance_id) != instance_id:
            raise ValueError("Registered ID differs from the running instance")
        capabilities = normalize_capabilities(cap_data.get("capabilities", cap_data))
        configuration = status_data.get("configuration") or {}
        if int(configuration.get("tensor_parallel_size", payload.get("num_gpus", 1))) != int(payload.get("num_gpus", 1)):
            raise ValueError("Registered GPU count differs from the running instance")
        advertised_range = status_data.get("context_range")
        if advertised_range is not None and tuple(advertised_range) != tuple(payload.get("context_range") or ()):
            raise ValueError("Registered context range differs from the running instance")
        backend_model = status_data.get("model", configuration.get("model"))
        for existing in self.instance_manager.get_instances_for_agent(payload["agent_id"]):
            if backend_model and existing.model and backend_model != existing.model:
                raise ValueError("All replicas of an agent must use the same model/tokenizer")
        info = self.instance_manager.register(
            instance_id, payload["agent_id"], payload.get("host", "127.0.0.1"),
            port=int(payload["port"]), context_range=payload.get("context_range"),
            num_gpus=int(payload.get("num_gpus", 1)), speculative_tokens=requested_k,
            capabilities=capabilities, model=backend_model,
            max_model_len=status_data.get("max_model_len", configuration.get("max_model_len")),
            backend_signature=status_data.get("backend_signature"),
            configuration=configuration)
        info.active_requests = int(status_data.get("active_requests", 0))
        self.instance_manager.set_ready(instance_id)
        return info

    def load_allocation(self, path):
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        default_gpus = int(data.get("config", {}).get("gpus_per_instance", 1))
        default_k = int(data.get("config", {}).get("speculative_tokens", 0))
        ordinal = 0
        for agent_id, allocation in data.get("allocation", {}).items():
            for group_index, group in enumerate(allocation.get("instances", [])):
                lower, upper = int(group["context_lower"]), int(group["context_upper"])
                for index in range(int(group.get("num_instances", 1))):
                    self.instance_manager.register(
                        instance_id=f"agent{ordinal}-g{group_index}-r{index}",
                        agent_id=agent_id, host="127.0.0.1", context_range=(lower, upper),
                        num_gpus=int(group.get("gpus_per_instance", default_gpus)),
                        speculative_tokens=int(group.get("speculative_tokens", group.get("speculative_k", default_k))))
                    ordinal += 1

    async def _dispatch_node_async(self, state, node):
        if state.error:
            return
        node.query_id = state.query_id
        scheduler = await self.get_scheduler(node.agent_id)
        if node.prompt is not None:
            # Use the actual tokenizer; text length/characters are not context tokens.
            infos = self.instance_manager.get_instances_for_agent(node.agent_id)
            if not infos:
                raise ValueError(f"No ready tokenizer instance for agent {node.agent_id}")
            tokenized = None
            for info in infos:
                try:
                    response = await self.http_client.post(
                        info.endpoint + "/tokenize", json={"prompt": node.prompt})
                    response.raise_for_status()
                    tokenized = response.json()
                    break
                except httpx.HTTPError:
                    continue
            if tokenized is None:
                raise ValueError(f"No healthy tokenizer instance for agent {node.agent_id}")
            node.input_tokens = token_ids(tokenized.get("token_ids", tokenized.get("prompt_token_ids")), "tokenizer result")
            if not node.input_tokens:
                raise ValueError("Tokenized prompt is empty")
            node.num_input_tokens = len(node.input_tokens)
        instance_id = await scheduler.route(state.query_id, node.invocation_id,
                                           node.input_tokens, node.num_input_tokens)
        if instance_id is None:
            node.status = NodeStatus.FAILED
            state.error = f"No healthy context-compatible instance for agent {node.agent_id}"
            return
        selected = self.instance_manager.get(instance_id)
        if state.input_semantics == "synthetic_lengths_ar" and selected.speculative_tokens != 0:
            raise ValueError("synthetic length replay is AR-only; never use it for MTP acceptance")
        for segment in node.input_segments or []:
            source_key = segment.get("parent", segment.get("parent_context"))
            if source_key is None:
                continue
            parent = state.query.dag.node_mapping[node_key(source_key)]
            source_model = (parent.result or {}).get("model")
            if not selected.model or not source_model or selected.model != source_model:
                raise ValueError("Live parent token bindings require identical known source/target models and tokenizers")
        payload = dict(request_id=f"q{state.query_id}_i{node.invocation_id}_r{node.repeat}",
                       agent_id=node.agent_id, workflow_id=str(state.query_id),
                       query_id=state.query_id, invocation_id=node.invocation_id,
                       repeat=node.repeat, max_tokens=node.max_tokens, temperature=0.0)
        payload.update(min_tokens=node.min_tokens, ignore_eos=node.ignore_eos,
                       input_semantics=state.input_semantics)
        payload["prompt_token_ids"] = node.input_tokens
        if node.cache_policy == "predictive" and not node.free_cache:
            if not selected.capabilities.get("predictive_cache"):
                raise ValueError("backend has no calibrated predictive cache policy")
            payload.update(cache_policy="predictive", tool_name=node.tool_name,
                           expected_tool_latency_seconds=node.expected_tool_latency_seconds)
        if node.keep_cache_in_gpu or not node.free_cache:
            needed = "retained_gpu_entries" if node.keep_cache_in_gpu else "cpu_swap"
            if not selected.capabilities.get(needed):
                raise ValueError(f"Backend does not implement requested {needed}")
            payload["keep_cache_in_gpu"] = node.keep_cache_in_gpu
            payload["free_cache"] = node.free_cache
        # Reference output IDs are never sent for forced decoding.
        result = await scheduler.dispatch(instance_id, payload)
        if not result or not result.get("success", False):
            node.status = NodeStatus.FAILED
            state.error = str((result or {}).get("error", "Generation failed"))
            return
        if result.get("request_id", payload["request_id"]) != payload["request_id"]:
            raise ValueError("Engine returned a mismatched request_id")
        normalized = dict(result)
        normalized["model"] = selected.model
        normalized["source_instance_id"] = instance_id
        normalized["output_token_ids"] = result.get("output_token_ids", result.get("token_ids", []))
        normalized["num_output_tokens"] = result.get("num_output_tokens", result.get("completion_tokens", len(normalized["output_token_ids"])))
        normalized["input_token_ids"] = result.get("input_token_ids", result.get("prompt_token_ids", node.input_tokens))
        normalized["num_input_tokens"] = result.get("num_input_tokens", result.get("prompt_tokens", len(normalized["input_token_ids"])))
        if int(normalized["num_input_tokens"]) != len(normalized["input_token_ids"]):
            raise ValueError("Engine input-token count mismatch")
        if not node.free_cache:
            normalized["cache_migration"] = await scheduler.maybe_migrate_request(
                source_instance_id=instance_id, query_id=state.query_id,
                invocation_id=node.invocation_id,
                input_token_ids=normalized["input_token_ids"],
                output_token_ids=normalized["output_token_ids"],
                keep_cache_in_gpu=(normalized.get("cache") or {}).get("keep_cache_in_gpu", node.keep_cache_in_gpu))
        elif any(info.capabilities.get("cache_state") for info in self.instance_manager.get_instances_for_agent(node.agent_id)):
            # A session may have owners in earlier partitions if a migration
            # was incompatible/deferred. Final release covers the whole agent,
            # not just whichever replica handled this last invocation.
            normalized["cache_release"] = await scheduler.release_request_cache(
                state.query_id, node.invocation_id, exclude_instance_id=instance_id)
        ready = state.query.get_next_execute_agents(
            node.invocation_id, state.query_id, node.repeat, result=normalized)
        await self._schedule_nodes_async(state, ready)
        if state.query.is_completed():
            state.completed = True

    async def _schedule_nodes_async(self, state, nodes):
        async def execute(node):
            try:
                await self._dispatch_node_async(state, node)
            except Exception as exc:
                node.status = NodeStatus.FAILED
                state.error = str(exc)
        await asyncio.gather(*(execute(node) for node in nodes))

    async def execute_query_async(self, state):
        try:
            await self._schedule_nodes_async(state, state.query.get_entry_nodes())
            state.completed = state.query.is_completed()
        except Exception as exc:
            state.error = str(exc)

    def start_query(self, state):
        task = asyncio.create_task(self.execute_query_async(state))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def close(self):
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._owns_http and self._http is not None:
            await self._http.aclose()


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    instance_id: str
    agent_id: str
    host: str = "127.0.0.1"
    port: int = Field(ge=1, le=65535)
    context_range: tuple[int, int] | None = None
    num_gpus: int = Field(default=1, ge=1)
    speculative_tokens: int = Field(default=0, ge=0)
    speculative_k: int | None = Field(default=None, ge=0)


class QueryRequest(BaseModel):
    query_id: int | None = None
    node_dependency: dict = Field(default_factory=dict)
    node_list: list = Field(default_factory=list)
    work_dir: str = "/tmp"
    input_semantics: str = "real_tokens"


def create_app(control: AsyncController | None = None):
    control = control or AsyncController()

    @asynccontextmanager
    async def lifespan(app):
        yield
        await control.close()

    app = FastAPI(lifespan=lifespan)
    app.state.controller = control

    @app.post("/register")
    async def register(req: RegisterRequest):
        payload = req.model_dump()
        if req.speculative_k is not None:
            if "speculative_tokens" in req.model_fields_set and req.speculative_tokens != req.speculative_k:
                raise HTTPException(422, "Conflicting fixed K fields")
            payload["speculative_tokens"] = req.speculative_k
        try:
            info = await control.register_instance(payload)
        except (ValueError, httpx.HTTPError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"status": "registered", "instance_id": info.instance_id,
                "speculative_tokens": info.speculative_tokens, "capabilities": info.capabilities}

    @app.post("/query")
    async def create_query(req: QueryRequest):
        query_id = req.query_id if req.query_id is not None else time.time_ns()
        try:
            query = Query(query_id, req.work_dir, req.node_dependency, req.node_list, time.time())
            if req.input_semantics not in {"real_tokens", "synthetic_lengths_ar"}:
                raise ValueError("unknown input semantics")
            state = QueryState(query_id, query, input_semantics=req.input_semantics)
            await control.add_query(state)
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        control.start_query(state)
        mode = ("synthetic_lengths_AR_only" if req.input_semantics == "synthetic_lengths_ar" else
                "live_token_dag" if any(n.input_segments is not None for n in query.dag.node_mapping.values()) else "fixed_history_replay")
        return {"status": "accepted", "query_id": query_id, "execution_mode": mode}

    @app.get("/query/{query_id}")
    async def query_status(query_id: int):
        state = await control.get_query(query_id)
        if state is None:
            raise HTTPException(404, "Unknown query")
        return {"query_id": query_id, "completed": state.completed, "error": state.error,
                "elapsed_seconds": (state.query.get_latency() if state.completed else time.time() - state.submit_time),
                "nodes": {f"{key[0]}:{key[1]}": {
                    "agent_id": node.agent_id, "status": node.status.name,
                    "input_token_ids": node.input_tokens, "output_token_ids": node.output_tokens,
                    "num_input_tokens": node.num_input_tokens,
                    "num_output_tokens": node.num_output_tokens,
                    "result": node.result,
                } for key, node in state.query.dag.node_mapping.items()}}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/instances")
    async def instances():
        return {"routing_policy": control.routing_policy, "instances": [
            {"instance_id": info.instance_id, "agent_id": info.agent_id, "model": info.model,
             "speculative_tokens": info.speculative_tokens, "ready": info.is_ready,
             "context_range": info.context_range, "endpoint": info.endpoint}
            for key in control.instance_manager.all_instance_ids()
            for info in [control.instance_manager.get(key)]]}

    @app.post("/tools/observe")
    async def observe_tool(req: ToolObservationRequest):
        destinations = [info for info in control.instance_manager.get_instances_for_agent(req.agent_id)
                        if info.capabilities.get("predictive_cache")]
        if not destinations:
            raise HTTPException(422, "no calibrated predictive-cache instance for this agent")
        results = {}
        for info in destinations:
            try:
                response = await control.http_client.post(info.endpoint + "/cache/tool_observation", json=req.model_dump())
                response.raise_for_status()
                results[info.instance_id] = response.json()
            except httpx.HTTPError as exc:
                results[info.instance_id] = {"error": str(exc)}
        return {"instances": results}

    return app


app = create_app()


def run_controller(host="127.0.0.1", port=5000, allocation_path=None, routing_policy="strict"):
    import uvicorn
    application = create_app(AsyncController(routing_policy=routing_policy))
    if allocation_path:
        application.state.controller.load_allocation(allocation_path)
    uvicorn.run(application, host=host, port=port)
