from __future__ import annotations

import argparse
import os
import subprocess
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
    parser = argparse.ArgumentParser(description="Run multiple GA/SA experiments and aggregate them.")
    parser.add_argument("--mode", choices=["synthetic", "real"], default="real")
    parser.add_argument("--preset", choices=["smoke", "quick", "standard"], default="quick")
    parser.add_argument("--seeds", default="42,7,13", help="Comma-separated random seeds.")
    parser.add_argument("--data-path", type=Path, default=Path("data/raw/train_transaction.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--search-features", type=int, default=None)
    parser.add_argument("--proxy-rows", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer seed.")

    completed: list[Path] = []
    for index, seed in enumerate(seeds, start=1):
        print(f"=== Seed {index}/{len(seeds)}: {seed} ===", flush=True)
        before = _run_dirs(args.results_dir)
        cmd = [
            sys.executable,
            "-u",
            "run_experiment.py",
            "--mode",
            args.mode,
            "--preset",
            args.preset,
            "--device",
            args.device,
            "--results-dir",
            str(args.results_dir),
            "--cache-dir",
            str(args.cache_dir),
            "--seed",
            str(seed),
        ]
        if args.mode == "real":
            cmd.extend(["--data-path", str(args.data_path)])
        if args.search_features is not None:
            cmd.extend(["--search-features", str(args.search_features)])
        if args.proxy_rows is not None:
            cmd.extend(["--proxy-rows", str(args.proxy_rows)])
        if args.no_cache:
            cmd.append("--no-cache")

        subprocess.run(cmd, check=True, env=_utf8_env())
        after = _run_dirs(args.results_dir)
        new_runs = sorted(after - before, key=lambda p: p.stat().st_mtime)
        if not new_runs:
            raise RuntimeError("Experiment finished but no new results/run_* directory was found.")
        completed.append(new_runs[-1])

    print("=== Aggregating completed runs ===", flush=True)
    out_dir = make_aggregate_dir(args.results_dir)
    aggregate_runs(completed, out_dir)
    print("Completed runs:", flush=True)
    for run_dir in completed:
        print(f"  {run_dir}", flush=True)
    print(f"Aggregate output: {out_dir}", flush=True)


def _run_dirs(results_dir: Path) -> set[Path]:
    if not results_dir.exists():
        return set()
    return {path.resolve() for path in results_dir.glob("run_*") if path.is_dir()}


def _utf8_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


if __name__ == "__main__":
    main()
