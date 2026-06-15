from __future__ import annotations

import json
import os
import sys
import types
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path.cwd()
RESULTS_DIR = ROOT / "results"
NOTEBOOK_SCRIPT = ROOT / "external_versions" / "ours_all_in_one.py"
EXPECTED_SEEDS = [7, 42, 2026]


def main() -> None:
    namespace = _load_notebook_core()
    run_single_experiment = namespace["run_single_experiment"]
    aggregate_runs = namespace["aggregate_runs"]
    write_aggregate_feature_overlap = namespace["write_aggregate_feature_overlap"]
    copy_figures_for_report = namespace["copy_figures_for_report"]
    write_chinese_aggregate_summary = namespace["write_chinese_aggregate_summary"]
    write_chinese_html_summary = namespace["write_chinese_html_summary"]

    data_path = namespace["DATA_PATH"]
    cache_dir = namespace["CACHE_DIR"]
    results_dir = namespace["RESULTS_DIR"]

    completed: list[Path] = []
    for seed in EXPECTED_SEEDS:
        existing = _find_latest_completed_run(seed)
        if existing is not None:
            print(f"Using existing word_strict seed {seed}: {existing}", flush=True)
            completed.append(existing)
            continue

        print("", flush=True)
        print(f"=== Running missing word_strict seed {seed} ===", flush=True)
        run_dir = run_single_experiment(
            mode="real",
            preset="word_strict",
            data_path=data_path,
            results_dir=results_dir,
            cache_dir=cache_dir,
            no_cache=False,
            device="cpu",
            proxy_rows=20_000,
            search_features=150,
            seed=seed,
            final_eval_full_training_set=True,
        )
        completed.append(Path(run_dir))

    out_dir = results_dir / f"aggregate_resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print("", flush=True)
    print("=== Aggregating completed word_strict seeds ===", flush=True)
    print(f"Aggregate output: {out_dir}", flush=True)
    aggregate_runs(completed, out_dir)
    write_aggregate_feature_overlap(completed, out_dir)
    representative_run = completed[-1]
    copy_figures_for_report(representative_run, out_dir)
    write_chinese_aggregate_summary(out_dir)
    write_chinese_html_summary(out_dir, representative_run=representative_run)

    print("", flush=True)
    print("Resume completed.", flush=True)
    print("Runs:", flush=True)
    for run_dir in completed:
        print(f"  {run_dir}", flush=True)
    print(f"Aggregate: {out_dir}", flush=True)
    print(f"HTML: {out_dir / 'GA_SA_report_summary_zh_TW.html'}", flush=True)


def _load_notebook_core() -> dict[str, object]:
    if not NOTEBOOK_SCRIPT.exists():
        raise FileNotFoundError(f"Missing notebook script: {NOTEBOOK_SCRIPT}")

    source = NOTEBOOK_SCRIPT.read_text(encoding="utf-8")
    marker = "\nRUN_SMOKE = False"
    marker_index = source.rfind(marker)
    if marker_index < 0:
        raise RuntimeError("Could not find final autorun marker in notebook script.")

    core_source = source[:marker_index]
    module_name = "timecare_notebook_core"
    module = types.ModuleType(module_name)
    module.__file__ = str(NOTEBOOK_SCRIPT)
    sys.modules[module_name] = module
    namespace = module.__dict__
    exec(compile(core_source, str(NOTEBOOK_SCRIPT), "exec"), namespace)
    for required in (
        "run_single_experiment",
        "aggregate_runs",
        "write_aggregate_feature_overlap",
        "copy_figures_for_report",
        "write_chinese_aggregate_summary",
        "write_chinese_html_summary",
    ):
        if required not in namespace:
            raise RuntimeError(f"Notebook core did not define {required}.")
    return namespace


def _find_latest_completed_run(seed: int) -> Path | None:
    if not RESULTS_DIR.exists():
        return None

    matches: list[Path] = []
    for run_dir in RESULTS_DIR.glob("run_*"):
        metadata_path = run_dir / "run_metadata.json"
        metrics_path = run_dir / "metrics.csv"
        selected_path = run_dir / "selected_features.json"
        if not metadata_path.exists() or not metrics_path.exists() or not selected_path.exists():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        config = metadata.get("config", {})
        if not isinstance(config, dict):
            continue
        if config.get("random_seed") != seed:
            continue
        if config.get("preset") != "word_strict":
            continue
        if metadata.get("final_evaluation_source") != "full_train_transaction":
            continue
        matches.append(run_dir)

    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


if __name__ == "__main__":
    main()
