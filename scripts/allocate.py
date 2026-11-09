"""Run the original DP/greedy structure with explicit measured MTP capacities.

This does not convert the old Llama profile into an MTP capacity model.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coreflow_mtp.allocator.capacity import CapacityTable, allocate_from_capacity_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity-table", required=True)
    parser.add_argument("--invocations", required=True, help="JSON mapping agent ID to average invocations per workflow")
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--allowed-k", type=int, nargs="+")
    parser.add_argument("--max-instance-group", type=int, default=3)
    parser.add_argument("--min-instance-group", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = allocate_from_capacity_table(CapacityTable.load(args.capacity_table), args.gpus,
            json.loads(Path(args.invocations).read_text()), args.max_instance_group,
            args.min_instance_group, args.allowed_k)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    path = Path(args.output)
    if path.exists():
        parser.exit(1, f"refusing to overwrite existing allocation: {path}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Wrote {path}; predicted workflow capacity={result['predicted_workflow_throughput']:.6g}/s")


if __name__ == "__main__":
    main()
