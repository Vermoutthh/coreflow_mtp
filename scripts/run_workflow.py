"""Submit a user-provided workflow DAG and wait for its actual result."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coreflow_mtp.launch import request_json


def wait_workflow(controller, payload, timeout=600):
    accepted = request_json(controller + "/query", payload)
    query_id = accepted["query_id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = request_json(f"{controller}/query/{query_id}")
        if result.get("error"):
            raise RuntimeError(result["error"])
        if result.get("completed"):
            return result
        time.sleep(.2)
    raise TimeoutError(f"workflow {query_id} did not complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-url", default="http://127.0.0.1:18500")
    parser.add_argument("--payload", required=True, help="semantic workflow JSON DAG")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", help="optional result JSON path")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    payload = json.loads(Path(args.payload).read_text())
    result = wait_workflow(args.controller_url.rstrip("/"), payload, timeout=args.timeout)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    for key, node in result["nodes"].items():
        generation = node.get("result") or {}
        print(f"{key} {node['agent_id']}: {generation.get('text', '')}")
    print(f"completed={result['completed']}, elapsed={result['elapsed_seconds']:.3f}s")


if __name__ == "__main__":
    main()
