"""Explicit AR-only length replay, never a semantic/MTP acceptance workload.

Graph dependencies and recorded lengths are preserved. Incremental input uses
marked filler text; reusable history binds to ancestors' actual generated
contexts. No fixed output-token list or logits processor is used.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from coreflow_mtp.allocator.request_analyzer import RequestAnalyzer
from coreflow_mtp.dag.dag import DAG, node_key
from coreflow_mtp.launch import request_json
from run_workflow import wait_workflow


def build_replay(records, dependencies, filler_ids, *, cache_policy="manual"):
    previous, nodes = [], []
    def ancestors(key):
        seen, pending = set(), list(dependencies.get(key, []))
        while pending:
            parent = pending.pop()
            if parent not in seen:
                seen.add(parent)
                pending.extend(dependencies.get(parent, []))
        return seen
    ordered = sorted(records, key=lambda item: node_key(item["node_id"]))
    last_repeat = {n["node_id"].split(":")[0]: n["node_id"] for n in ordered}
    for record in ordered:
        incremental, output, cached = RequestAnalyzer._token_counts(record)
        key, agent = record["node_id"], record["agent_name"]
        if output < 1 or incremental < 0 or cached < 0:
            raise ValueError("length replay requires nonnegative inputs and positive outputs")
        segments = []
        if cached:
            possible = [p for p in previous if p["id"] in ancestors(key) and p["total"] >= cached]
            possible.sort(key=lambda p: (p["agent"] == agent, p["total"] == cached,
                                         node_key(p["id"])), reverse=True)
            if not possible:
                raise ValueError(f"{key}: no actual dependency ancestor supplies the declared reusable prefix")
            segments.append({"parent_context": possible[0]["id"], "take": cached})
        if incremental:
            filler = (filler_ids * ((incremental + len(filler_ids) - 1) // len(filler_ids)))[:incremental]
            segments.append({"token_ids": filler})
        final = last_repeat[key.split(":")[0]] == key
        nodes.append({"node_id": key, "agent_name": agent, "input_segments": segments,
            "max_tokens": output, "min_tokens": output, "ignore_eos": True,
            "free_cache": final, "keep_cache_in_gpu": not final and cache_policy == "manual",
            "cache_policy": cache_policy})
        previous.append({"id": key, "agent": agent, "total": incremental + cached + output})
    payload = {"input_semantics": "synthetic_lengths_ar", "node_dependency": dependencies, "node_list": nodes}
    dag = DAG()
    dag.init_from_payload(dependencies, nodes)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-list", required=True)
    parser.add_argument("--dependencies", required=True)
    parser.add_argument("--query-key")
    parser.add_argument("--model", required=True)
    parser.add_argument("--controller-url", required=True)
    parser.add_argument("--allow-synthetic-lengths", action="store_true")
    parser.add_argument("--cache-policy", choices=["manual", "predictive"], default="manual")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.allow_synthetic_lengths:
        parser.error("length traces have no real semantics; explicit --allow-synthetic-lengths is required")
    records = json.loads(Path(args.node_list).read_text())
    dependencies = json.loads(Path(args.dependencies).read_text())
    key = args.query_key or next(iter(records))
    agents = {n["agent_name"] for n in records[key]}
    state = request_json(args.controller_url.rstrip("/") + "/instances")
    selected = [info for info in state["instances"] if info["agent_id"] in agents and info["ready"]]
    if ({i["agent_id"] for i in selected} != agents or
            any(i["speculative_tokens"] != 0 or Path(i["model"]).resolve() != Path(args.model).resolve() for i in selected)):
        parser.error("every required agent must have a ready matching-model K=0 instance")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    filler = tokenizer.encode(" The Earth rotates and daylight changes with time.", add_special_tokens=False)
    payload = build_replay(records[key], dependencies[key], filler, cache_policy=args.cache_policy)
    output = Path(args.output)
    if output.exists():
        parser.error("refusing to overwrite a previous replay")
    result = wait_workflow(args.controller_url.rstrip("/"), payload, timeout=3600)
    for record in records[key]:
        node = result["nodes"][record["node_id"]]
        incremental, requested, cached = RequestAnalyzer._token_counts(record)
        if node["num_input_tokens"] != incremental + cached or node["num_output_tokens"] != requested:
            raise AssertionError("native replay did not preserve recorded lengths")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"input_semantics": "synthetic_lengths_AR_only_not_MTP_acceptance",
        "source_node_list": str(Path(args.node_list).resolve()), "query_key": key,
        "payload": payload, "workflow": result}, ensure_ascii=False, indent=2))
    print(f"AR-only length replay completed: {len(result['nodes'])} nodes -> {output}")


if __name__ == "__main__":
    main()
