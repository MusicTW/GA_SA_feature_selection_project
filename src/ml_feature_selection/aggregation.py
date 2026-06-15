from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


def make_aggregate_dir(results_dir: Path) -> Path:
    out_dir = results_dir / f"aggregate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def aggregate_runs(run_dirs: Iterable[Path], out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, object]] = []

    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            continue

        metrics = pd.read_csv(metrics_path)
        metadata = _read_metadata(run_dir)
        config = metadata.get("config", {}) if isinstance(metadata.get("config"), dict) else {}
        data = metadata.get("data", {}) if isinstance(metadata.get("data"), dict) else {}

        metrics["run_dir"] = str(run_dir)
        metrics["preset"] = config.get("preset", "unknown")
        metrics["seed"] = config.get("random_seed", "unknown")
        metrics["mode"] = metadata.get("mode", "unknown")
        metrics["device"] = metadata.get("device", "unknown")
        metrics["rows"] = data.get("rows", None)
        metrics["search_features"] = data.get("search_features", None)
        rows.append(metrics)

        metadata_rows.append(
            {
                "run_dir": str(run_dir),
                "preset": config.get("preset", "unknown"),
                "seed": config.get("random_seed", "unknown"),
                "mode": metadata.get("mode", "unknown"),
                "device": metadata.get("device", "unknown"),
                "rows": data.get("rows", None),
                "search_features": data.get("search_features", None),
            }
        )

    if not rows:
        raise FileNotFoundError("No metrics.csv files found in the supplied run directories.")

    all_metrics = pd.concat(rows, ignore_index=True)
    group_cols = ["lambda", "method"]
    summary = (
        all_metrics.groupby(group_cols, dropna=False)
        .agg(
            n_runs=("run_dir", "nunique"),
            final_cv_auc_mean=("final_cv_auc", "mean"),
            final_cv_auc_std=("final_cv_auc", "std"),
            final_cv_fitness_mean=("final_cv_fitness", "mean"),
            final_cv_fitness_std=("final_cv_fitness", "std"),
            selected_count_mean=("selected_count", "mean"),
            selected_count_std=("selected_count", "std"),
            inner_auc_mean=("auc", "mean"),
            inner_auc_std=("auc", "std"),
            seconds_mean=("seconds", "mean"),
        )
        .reset_index()
        .sort_values(["lambda", "final_cv_fitness_mean"], ascending=[True, False])
    )
    summary = summary.fillna(
        {
            "final_cv_auc_std": 0.0,
            "final_cv_fitness_std": 0.0,
            "selected_count_std": 0.0,
            "inner_auc_std": 0.0,
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    all_metrics.to_csv(out_dir / "all_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "aggregate_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metadata_rows).to_csv(out_dir / "runs.csv", index=False, encoding="utf-8-sig")
    _write_summary_md(out_dir, summary, pd.DataFrame(metadata_rows))
    return all_metrics, summary


def _read_metadata(run_dir: Path) -> dict[str, object]:
    path = run_dir / "run_metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_summary_md(out_dir: Path, summary: pd.DataFrame, runs: pd.DataFrame) -> None:
    full_seed_count = int(summary["n_runs"].max())
    lines = [
        "# Multi-Seed Aggregate Summary",
        "",
        "## Runs",
        "",
        "```text",
        runs.to_string(index=False),
        "```",
        "",
        "## Best Method Per Lambda",
        "",
        f"Only methods present in all `{full_seed_count}` runs are considered here. "
        "Rows with fewer runs are still listed in the full metric table below.",
        "",
    ]

    for lam, group in summary.groupby("lambda", sort=True):
        comparable = group[group["n_runs"] == full_seed_count]
        if comparable.empty:
            comparable = group
        best_auc = comparable.sort_values("final_cv_auc_mean", ascending=False).iloc[0]
        best_fitness = comparable.sort_values("final_cv_fitness_mean", ascending=False).iloc[0]
        lines.append(
            f"- lambda `{lam}`: best mean AUC = `{best_auc['method']}` "
            f"({best_auc['final_cv_auc_mean']:.4f} +/- {best_auc['final_cv_auc_std']:.4f}, "
            f"features {best_auc['selected_count_mean']:.1f}); "
            f"best mean fitness = `{best_fitness['method']}` "
            f"({best_fitness['final_cv_fitness_mean']:.4f} +/- {best_fitness['final_cv_fitness_std']:.4f}, "
            f"features {best_fitness['selected_count_mean']:.1f})"
        )

    lines.extend(
        [
            "",
            "## Aggregate Metrics",
            "",
            "```text",
            summary.to_string(index=False),
            "```",
            "",
            "## Files",
            "",
            "- `all_metrics.csv`: every metric row from every run",
            "- `aggregate_metrics.csv`: mean/std by lambda and method",
            "- `runs.csv`: run directory and metadata",
        ]
    )
    (out_dir / "aggregate_summary.md").write_text("\n".join(lines), encoding="utf-8")
