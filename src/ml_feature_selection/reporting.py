from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from .config import ExperimentConfig
from .data import PreparedData


def make_run_dir(results_dir: Path) -> Path:
    from datetime import datetime

    run_dir = results_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    (run_dir / "plots").mkdir(parents=True, exist_ok=False)
    return run_dir


def write_outputs(
    run_dir: Path,
    config: ExperimentConfig,
    data: PreparedData,
    metrics: list[dict[str, object]],
    selected: dict[str, list[str]],
    convergence: list[dict[str, object]],
    extra_metadata: dict[str, object] | None = None,
) -> None:
    metrics_df = pd.DataFrame(metrics)
    convergence_df = pd.DataFrame(convergence)

    metrics_df.to_csv(run_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    convergence_df.to_csv(run_dir / "convergence.csv", index=False, encoding="utf-8-sig")
    (run_dir / "selected_features.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "config": asdict(config),
        "data": {
            "source": data.source,
            "rows": data.rows,
            "original_features": data.original_features,
            "search_features": len(data.feature_names),
        },
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _plot_convergence(convergence_df, run_dir / "plots" / "convergence.png")
    _plot_pareto(metrics_df, run_dir / "plots" / "pareto_auc_vs_features.png")
    _plot_lambda(metrics_df, run_dir / "plots" / "lambda_sensitivity.png")
    _plot_feature_selection_by_mi_rank(
        data.feature_names,
        selected,
        run_dir / "plots" / "feature_importance_selection.png",
    )
    _write_feature_overlap(run_dir, metrics_df, selected)
    _write_summary(run_dir, config, data, metrics_df)


def selected_feature_names(feature_names: list[str], mask: Iterable[bool]) -> list[str]:
    return [name for name, keep in zip(feature_names, mask) if keep]


def _plot_convergence(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    plt.figure(figsize=(10, 6))
    for (method, lam), group in df.groupby(["method", "lambda"]):
        if method not in {"GA", "SA"}:
            continue
        plt.plot(group["step"], group["best_fitness"], label=f"{method} best lambda={lam}")
        if method == "GA" and "avg_or_current_fitness" in group:
            plt.plot(
                group["step"],
                group["avg_or_current_fitness"],
                linestyle="--",
                alpha=0.65,
                label=f"GA avg lambda={lam}",
            )
    plt.xlabel("Step")
    plt.ylabel("Fitness")
    plt.title("GA/SA convergence: best and GA average fitness")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_pareto(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    auc_col = "final_cv_auc" if "final_cv_auc" in df.columns else "auc"
    plt.figure(figsize=(9, 6))
    for method, group in df.groupby("method"):
        plt.scatter(group["selected_count"], group[auc_col], label=method, s=60)
    plt.xlabel("Selected feature count")
    plt.ylabel("AUC")
    plt.title("AUC vs selected feature count")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_lambda(df: pd.DataFrame, path: Path) -> None:
    main = df[df["method"].isin(["GA", "SA", "RandomSearch"])]
    if main.empty:
        return
    auc_col = "final_cv_auc" if "final_cv_auc" in main.columns else "auc"
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    for method, group in main.groupby("method"):
        group = group.sort_values("lambda")
        ax1.plot(group["lambda"], group[auc_col], marker="o", label=f"{method} AUC")
        ax2.plot(
            group["lambda"],
            group["selected_count"],
            marker="x",
            linestyle="--",
            label=f"{method} features",
        )
    ax1.set_xlabel("lambda")
    ax1.set_ylabel("AUC")
    ax2.set_ylabel("Selected features")
    ax1.grid(alpha=0.25)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    plt.title("Lambda sensitivity")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_feature_selection_by_mi_rank(
    feature_names: list[str],
    selected: dict[str, list[str]],
    path: Path,
    top_n: int = 30,
) -> None:
    if not feature_names or not selected:
        return

    lambdas: list[str] = []
    for key in selected:
        if "::" not in key or "::GA" not in key:
            continue
        lam = key.split("::", 1)[0].replace("lambda=", "")
        lambdas.append(lam)
    preferred = [lam for lam in ["0.05", "0.1", "0.10"] if lam in lambdas]
    chosen_lambdas = preferred[:2] if preferred else sorted(set(lambdas))[:2]
    if not chosen_lambdas:
        return

    top_features = feature_names[:top_n]
    rank_score = list(range(len(top_features), 0, -1))
    display_features = [f"{idx + 1:02d}. {name}" for idx, name in enumerate(top_features)]

    columns: list[tuple[str, set[str]]] = []
    for lam in chosen_lambdas:
        for method in ("GA", "SA"):
            values = selected.get(f"lambda={lam}::{method}")
            if values is not None:
                columns.append((f"{method} {lam}", set(values)))
    if not columns:
        return

    matrix = [
        [1 if feature in feature_set else 0 for _, feature_set in columns]
        for feature in top_features
    ]

    fig, (ax_bar, ax_sel) = plt.subplots(
        ncols=2,
        figsize=(12, 9),
        gridspec_kw={"width_ratios": [2.2, 1.2]},
        sharey=True,
    )
    y_pos = list(range(len(top_features)))[::-1]
    ax_bar.barh(y_pos, rank_score, color="#4c78a8", alpha=0.86)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(display_features[::-1], fontsize=9)
    ax_bar.set_xlabel("Mutual information rank score")
    ax_bar.set_title("Top MI-ranked features")
    ax_bar.grid(axis="x", alpha=0.22)

    ax_sel.imshow(matrix[::-1], aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax_sel.set_xticks(range(len(columns)))
    ax_sel.set_xticklabels([label for label, _ in columns], rotation=35, ha="right")
    ax_sel.set_yticks(y_pos)
    ax_sel.tick_params(axis="y", left=False, labelleft=False)
    ax_sel.set_title("Selected by method")
    for row, values in enumerate(matrix[::-1]):
        for col, value in enumerate(values):
            if value:
                ax_sel.text(col, row, "yes", ha="center", va="center", fontsize=8)

    fig.suptitle("Feature importance rank and GA/SA selection", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(path, dpi=160)
    plt.close()


def _write_feature_overlap(
    run_dir: Path,
    metrics_df: pd.DataFrame,
    selected: dict[str, list[str]],
) -> None:
    rows: list[dict[str, float | int]] = []
    for lam in sorted(metrics_df["lambda"].dropna().unique()):
        lam_text = str(lam)
        ga = set(selected.get(f"lambda={lam_text}::GA", []))
        sa = set(selected.get(f"lambda={lam_text}::SA", []))
        if not ga or not sa:
            continue
        overlap = len(ga & sa)
        union = len(ga | sa)
        rows.append(
            {
                "lambda": float(lam),
                "ga_features": len(ga),
                "sa_features": len(sa),
                "overlap": overlap,
                "jaccard_similarity": overlap / union if union else 0.0,
            }
        )
    if rows:
        pd.DataFrame(rows).to_csv(
            run_dir / "feature_overlap_ga_sa.csv",
            index=False,
            encoding="utf-8-sig",
        )


def _write_summary(
    run_dir: Path,
    config: ExperimentConfig,
    data: PreparedData,
    metrics_df: pd.DataFrame,
) -> None:
    score_col = "final_cv_fitness" if "final_cv_fitness" in metrics_df.columns else "fitness"
    auc_col = "final_cv_auc" if "final_cv_auc" in metrics_df.columns else "auc"
    best_rows = metrics_df.sort_values(["lambda", score_col], ascending=[True, False])
    lines = [
        "# GA/SA Feature Selection Run Summary",
        "",
        f"- preset: `{config.preset}`",
        f"- source: `{data.source}`",
        f"- rows: `{data.rows}`",
        f"- original features: `{data.original_features}`",
        f"- search features after preprocessing/MI: `{len(data.feature_names)}`",
        f"- inner validation folds: `{config.inner_cv_folds}` (`1` means holdout)",
        f"- final CV folds: `{config.final_cv_folds}`",
        "",
        "## Best Method Per Lambda",
        "",
    ]
    for lam, group in best_rows.groupby("lambda", sort=True):
        row = group.iloc[0]
        lines.append(
            f"- lambda `{lam}`: `{row['method']}` "
            f"final AUC=`{row[auc_col]:.4f}`, final fitness=`{row[score_col]:.4f}`, "
            f"features=`{int(row['selected_count'])}`"
        )

    lines.extend(
        [
            "",
            "## All Metrics",
            "",
            "```text",
            metrics_df.to_string(index=False),
            "```",
            "",
            "## Files",
            "",
            "- `metrics.csv`",
            "- `selected_features.json`",
            "- `convergence.csv`",
            "- `plots/convergence.png`",
            "- `plots/pareto_auc_vs_features.png`",
            "- `plots/lambda_sensitivity.png`",
            "- `plots/feature_importance_selection.png`",
            "- `feature_overlap_ga_sa.csv`",
        ]
    )
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
