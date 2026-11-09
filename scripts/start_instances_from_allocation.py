"""Isolated wrapper for the allocation-driven native MTP launcher."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coreflow_mtp.launch import main

if __name__ == "__main__":
    main()
