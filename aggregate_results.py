from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from ml_feature_selection.aggregation import aggregate_runs, make_aggregate_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multiple GA/SA experiment runs.")
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Run directories under results/.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or make_aggregate_dir(args.results_dir)
    all_metrics, summary = aggregate_runs(args.run_dirs, out_dir)
    print(f"Aggregated rows: {len(all_metrics)}")
    print(f"Summary rows: {len(summary)}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
