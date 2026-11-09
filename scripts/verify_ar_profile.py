import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coreflow_mtp.profiler.verification import verify_profile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K=0组件拟合与留context验证，不启动GPU")
    parser.add_argument("--profile-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    verify_profile(args.profile_bundle, args.output_dir)
    print(Path(args.output_dir) / "拟合检查_zh.md")
