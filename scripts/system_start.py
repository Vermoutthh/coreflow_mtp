"""Start the migrated controller without importing the old repository."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coreflow_mtp.controller import run_controller


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18500)
    parser.add_argument("--allocation", help="optional planned-instance stubs; live registration still verifies the backend")
    parser.add_argument("--routing-policy", choices=["strict", "coreflow"], default="strict")
    args = parser.parse_args()
    run_controller(args.host, args.port, args.allocation, args.routing_policy)


if __name__ == "__main__":
    main()
