"""Profile original K=0 CoreFlow components using installed vLLM 0.18.1."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="pinned local Llama checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--blocks", type=int, default=100000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--grid", help="explicit JSON grid; saved unchanged in the result provenance")
    parser.add_argument("--timing-protocol", choices=["coreflow_forward", "prepared_kernel"], default="coreflow_forward")
    args = parser.parse_args()
    from coreflow_mtp.profiler.runner import run_profile
    run_profile(args.model, args.output_dir, device=args.device, warmup=args.warmup,
        repeat=args.repeat, blocks=args.blocks, smoke=args.smoke, timing_protocol=args.timing_protocol,
        grid=json.loads(Path(args.grid).read_text()) if args.grid else None)


if __name__ == "__main__":
    main()
