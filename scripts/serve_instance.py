"""Use: <mtp-python> -I -B scripts/serve_instance.py --help."""
import sys
from pathlib import Path

# -I deliberately excludes both inherited PYTHONPATH and this script's folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coreflow_mtp.runtime.instance import main

if __name__ == "__main__":
    main()
