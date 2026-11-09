"""Validated trace DAG with real generated results and optional live bindings.

Legacy full-input traces are fixed-history replay, not live tool execution.
For a live token DAG use input_segments containing token_ids or parent node IDs.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import time


class NodeStatus(Enum):
    WAITING = auto()
    EXECUTING = auto()
    FINISHED = auto()
    FAILED = auto()


def node_key(value):
    if isinstance(value, str):
        parts = value.split(":")
        if len(parts) != 2:
            raise ValueError("node_id must be invocation_id:repeat")
        value = tuple(map(int, parts))
    if not isinstance(value, (list, tuple)) or len(value) != 2 or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in value):
        raise ValueError("node_id requires exactly two nonnegative integers")
    return tuple(value)


def token_ids(value, field_name="token_ids"):
    if not isinstance(value, (list, tuple)) or any(
            not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in value):
        raise ValueError(f"{field_name} must be nonnegative integer token IDs")
    return list(value)


@dataclass
class Node:
    agent_id: str
    invocation_id: int
    repeat: int = 0
    status: NodeStatus = NodeStatus.WAITING
    query_id: int = 0
    input_tokens: list[int] = field(default_factory=list)
    reference_output_tokens: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)
    prompt: str | None = None
    input_segments: list[dict] | None = None
    num_input_tokens: int = 0
    num_output_tokens: int = 0
    max_tokens: int = 128
    min_tokens: int = 0
    ignore_eos: bool = False
    cached_tokens: int = 0
    keep_cache_in_gpu: bool = False
    free_cache: bool = True
    cache_policy: str = "manual"
    tool_name: str | None = None
    expected_tool_latency_seconds: float | None = None
    start_time: float = 0.0
    end_time: float | None = None
    result: dict | None = None

    @property
    def key(self):
        return self.invocation_id, self.repeat


class DAG:
    def __init__(self, work_dir=None):
        self.workflow: dict[tuple[int, int], list[tuple[int, int]]] = {}
        self.node_mapping: dict[tuple[int, int], Node] = {}
        self.excuted_node = set()  # Historical spelling retained for consumers.
        self.start_time = time.time()
        self.end_time = None
        self.end_to_end_latency = None

    @property
    def node_count(self):
        return len(self.node_mapping)

    def init_from_payload(self, node_dependency: dict, node_list: list):
        if not node_list:
            raise ValueError("Workflow must contain at least one semantic LLM node")
        for record in node_list:
            key = node_key(record["node_id"])
            if key in self.node_mapping:
                raise ValueError(f"Duplicate node: {key}")
            ids = record.get("input_token_ids", record.get("prompt_token_ids"))
            prompt = record.get("prompt")
            segments = record.get("input_segments")
            if prompt is not None and (not isinstance(prompt, str) or not prompt):
                raise ValueError("prompt must be a nonempty string")
            if record.get("keep_cache_in_gpu") is True and record.get("free_cache", True):
                raise ValueError("keep_cache_in_gpu requires free_cache=false")
            policy = record.get("cache_policy", "manual")
            if policy not in {"manual", "predictive"}:
                raise ValueError("unknown cache policy")
            if policy == "predictive" and record.get("keep_cache_in_gpu"):
                raise ValueError("predictive policy decides whether to keep GPU cache")
            expected = record.get("expected_tool_latency_seconds")
            if expected is not None and (not isinstance(expected, (int, float))
                    or not math.isfinite(expected) or expected < 0):
                raise ValueError("invalid expected tool latency")
            if sum(value is not None for value in (ids, prompt, segments)) != 1:
                raise ValueError("Each node requires exactly one of real input_token_ids, prompt or input_segments; length-only traces cannot run MTP")
            values = token_ids(ids, "input_token_ids") if ids is not None else []
            if ids is not None and not values:
                raise ValueError("Empty prompt token list")
            declared = record.get("num_input_tokens")
            if ids is not None and declared is not None and int(declared) != len(values):
                raise ValueError("num_input_tokens differs from full input_token_ids length")
            reference = token_ids(record.get("output_token_ids", []), "output_token_ids")
            budget = record.get("max_tokens", record.get("num_output_tokens"))
            if budget is None:
                budget = len(reference) or 128
            if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
                raise ValueError("max_tokens must be positive")
            minimum = record.get("min_tokens", 0)
            if type(minimum) is not int or not 0 <= minimum <= budget:
                raise ValueError("min_tokens must lie within the output budget")
            node = Node(agent_id=str(record.get("agent_name", record.get("agent_id", ""))),
                        invocation_id=key[0], repeat=key[1], input_tokens=values,
                        prompt=prompt, input_segments=segments,
                        reference_output_tokens=reference,
                        num_input_tokens=len(values), max_tokens=int(budget),
                        min_tokens=minimum, ignore_eos=bool(record.get("ignore_eos", False)),
                        cached_tokens=int(record.get("cached_tokens", 0)),
                        keep_cache_in_gpu=bool(record.get("keep_cache_in_gpu", False)),
                        free_cache=bool(record.get("free_cache", True)), cache_policy=policy,
                        tool_name=record.get("tool_name"), expected_tool_latency_seconds=expected)
            if not node.agent_id:
                raise ValueError("Missing agent ID")
            self.node_mapping[key] = node
        self.workflow = {key: [] for key in self.node_mapping}
        for child, parents in node_dependency.items():
            key = node_key(child)
            if key not in self.node_mapping:
                raise ValueError(f"Unknown dependency child: {child}")
            self.workflow[key] = [node_key(parent) for parent in parents]
            if any(parent not in self.node_mapping for parent in self.workflow[key]):
                raise ValueError(f"Unknown dependency parent: {child}")
        for key, node in self.node_mapping.items():
            for segment in node.input_segments or []:
                if set(segment) == {"parent"}:
                    if node_key(segment["parent"]) not in self.workflow[key]:
                        raise ValueError("input_segments parent must be a direct dependency")
                elif set(segment) == {"token_ids"}:
                    token_ids(segment["token_ids"])
                elif "parent_context" in segment and set(segment) <= {"parent_context", "take"}:
                    ancestor = node_key(segment["parent_context"])
                    reachable, pending = set(), list(self.workflow[key])
                    while pending:
                        candidate = pending.pop()
                        if candidate not in reachable:
                            reachable.add(candidate)
                            pending.extend(self.workflow[candidate])
                    if ancestor == key or ancestor not in reachable:
                        raise ValueError("parent_context must reference a real dependency ancestor")
                    if "take" in segment and (type(segment["take"]) is not int or segment["take"] < 0):
                        raise ValueError("parent_context take must be nonnegative")
                else:
                    raise ValueError("input_segments supports token_ids, parent or ancestor parent_context")
        # Kahn traversal rejects cycles before execution rather than hanging.
        reached = set()
        while True:
            next_keys = {key for key, deps in self.workflow.items()
                         if key not in reached and set(deps) <= reached}
            if not next_keys:
                break
            reached.update(next_keys)
        if len(reached) != self.node_count:
            raise ValueError("DAG contains a dependency cycle")

    def resolve_input(self, node: Node):
        if node.input_segments is not None:
            result = []
            for segment in node.input_segments:
                if "token_ids" in segment:
                    result.extend(segment["token_ids"])
                elif "parent" in segment:
                    parent = self.node_mapping[node_key(segment["parent"])]
                    if parent.status != NodeStatus.FINISHED:
                        raise ValueError("Parent result is not ready")
                    result.extend(parent.output_tokens)
                else:
                    parent = self.node_mapping[node_key(segment["parent_context"])]
                    if parent.status != NodeStatus.FINISHED:
                        raise ValueError("Ancestor context is not ready")
                    context = parent.input_tokens + parent.output_tokens
                    take = segment.get("take", len(context))
                    if take > len(context):
                        raise ValueError("requested cached prefix exceeds the ancestor's actual context")
                    result.extend(context[:take])
            if not result:
                raise ValueError("Resolved prompt is empty")
            node.input_tokens = result
            node.num_input_tokens = len(result)
        return node.input_tokens

    def _ready(self):
        nodes = []
        for key, deps in self.workflow.items():
            node = self.node_mapping[key]
            if node.status == NodeStatus.WAITING and set(deps) <= self.excuted_node:
                self.resolve_input(node)
                node.status = NodeStatus.EXECUTING
                node.start_time = time.time()
                nodes.append(node)
        return nodes

    def get_entry_nodes(self):
        return self._ready()

    def update_result(self, invocation_id, query_id, repeat, result=None):
        node = self.node_mapping[(invocation_id, repeat)]
        if node.status == NodeStatus.FINISHED:
            return []
        if node.status != NodeStatus.EXECUTING:
            raise ValueError("Cannot complete a node that is not executing")
        if result is not None:
            generated = token_ids(result.get("output_token_ids", []), "output_token_ids")
            if int(result.get("num_output_tokens", len(generated))) != len(generated):
                raise ValueError("Engine output-token count mismatch")
            node.output_tokens = generated
            node.num_output_tokens = len(generated)
            inputs = result.get("input_token_ids")
            if inputs is not None:
                node.input_tokens = token_ids(inputs, "input_token_ids")
                node.num_input_tokens = len(node.input_tokens)
            node.result = dict(result)
        node.status = NodeStatus.FINISHED
        node.end_time = time.time()
        self.excuted_node.add(node.key)
        ready = self._ready()
        if self.is_completed():
            self.end_time = time.time()
            self.end_to_end_latency = self.end_time - self.start_time
        return ready

    def is_completed(self):
        return len(self.excuted_node) == self.node_count


def create_dag_by_json(work_dir, node_dependency, node_list):
    dag = DAG(work_dir)
    dag.init_from_payload(node_dependency, node_list)
    return dag
