from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGURE_NAMES = [
    "01_convergence.png",
    "02_feature_importance_selection.png",
    "03_pareto_auc_vs_features.png",
    "04_lambda_sensitivity.png",
]

FIGURE_CAPTIONS = [
    "收斂曲線比較：seed 42 的代表性 run。x 軸正規化為搜尋進度百分比，方便比較 GA 60 generations 與 SA 5000 iterations 的搜尋趨勢。",
    "特徵重要性與選取狀態：seed 42 的代表性 run。左側為 Mutual Information 前 30 名特徵；右側色塊代表 GA/SA 是否選中該特徵。",
    "Pareto Front：seed 42 的代表性 run。呈現 final CV AUC 與選取特徵數的取捨，標示 GA、SA、Random Search、All Features 與 Top-K MI baseline。",
    "Lambda 敏感度分析：seed 42 的代表性 run。比較 λ 增加時，AUC 與選取特徵數如何變化。",
]


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft JhengHei",
        "Microsoft YaHei",
        "Noto Sans CJK TC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120


def find_seed_run(aggregate_dir: Path, seed: int) -> Path:
    runs = pd.read_csv(aggregate_dir / "runs.csv", encoding="utf-8-sig")
    match = runs[runs["seed"].astype(int) == seed]
    if match.empty:
        raise RuntimeError(f"Cannot find seed {seed} in {aggregate_dir / 'runs.csv'}")
    run_dir = Path(str(match.iloc[0]["run_dir"]))
    if not run_dir.is_absolute():
        run_dir = Path.cwd() / run_dir
    return run_dir


def format_lambda(value: float) -> str:
    if abs(value) < 1e-12:
        return "0"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def selected_key(lam: float, method: str) -> str:
    return f"lambda={str(lam)}::{method}"


def plot_convergence(run_dir: Path, fig_dir: Path, seed: int) -> None:
    conv = pd.read_csv(run_dir / "convergence.csv", encoding="utf-8-sig")
    grid = np.linspace(0.0, 100.0, 101)
    lambdas = [0.0, 0.05, 0.1, 0.2]
    series_specs = [
        ("GA", "best_fitness", "GA best fitness", "#1f77b4", "-"),
        ("GA", "avg_or_current_fitness", "GA average fitness", "#ff7f0e", "--"),
        ("SA", "best_fitness", "SA best fitness", "#2ca02c", "-"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    axes = axes.ravel()
    handles = []
    labels = []
    for ax, lam in zip(axes, lambdas):
        for method, column, label, color, linestyle in series_specs:
            group = conv[(conv["method"] == method) & (np.isclose(conv["lambda"], lam))]
            if group.empty or column not in group.columns:
                continue
            group = group.sort_values("step")
            max_step = float(group["step"].max())
            progress = group["step"].astype(float).to_numpy() / max_step * 100.0
            y = group[column].astype(float).to_numpy()
            y_interp = np.interp(grid, progress, y, left=y[0], right=y[-1])
            (line,) = ax.plot(grid, y_interp, label=label, color=color, linestyle=linestyle, linewidth=2.1)
            if label not in labels:
                handles.append(line)
                labels.append(label)
        ax.set_title(f"λ = {format_lambda(lam)}", fontsize=13)
        ax.set_xlabel("搜尋進度 (%)")
        ax.set_ylabel("Fitness")
        ax.grid(alpha=0.25)

    fig.suptitle(f"圖一：GA/SA 收斂曲線比較（seed {seed}）", fontsize=17)
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.text(
        0.01,
        0.01,
        "說明：此圖用代表性 seed 展示搜尋過程；multi-seed 穩定性請看彙整表的 mean/std。",
        fontsize=10,
        color="#3e4c59",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    fig.savefig(fig_dir / "01_convergence.png", dpi=180)
    plt.close(fig)


def plot_feature_selection(run_dir: Path, fig_dir: Path, seed: int, top_n: int = 30) -> None:
    selected = json.loads((run_dir / "selected_features.json").read_text(encoding="utf-8"))
    feature_order = selected.get("lambda=0.0::AllSearchFeatures")
    if not feature_order:
        raise RuntimeError("Cannot find lambda=0.0::AllSearchFeatures for MI feature order.")
    top_features = list(feature_order[:top_n])
    columns = [(0.05, "GA"), (0.05, "SA"), (0.1, "GA"), (0.1, "SA")]
    matrix = np.zeros((top_n, len(columns)), dtype=int)

    for col_idx, (lam, method) in enumerate(columns):
        feature_set = set(selected.get(selected_key(lam, method), []))
        for row_idx, feature in enumerate(top_features):
            matrix[row_idx, col_idx] = int(feature in feature_set)

    rank_score = np.arange(top_n, 0, -1)
    labels = [f"{i + 1:02d}. {feature}" for i, feature in enumerate(top_features)]
    y = np.arange(top_n)

    fig, (ax_rank, ax_sel) = plt.subplots(
        ncols=2,
        figsize=(14, 10),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
        sharey=True,
    )
    ax_rank.barh(y, rank_score, color="#4c78a8", alpha=0.88)
    ax_rank.set_yticks(y)
    ax_rank.set_yticklabels(labels, fontsize=9)
    ax_rank.invert_yaxis()
    ax_rank.set_xlabel("MI rank score")
    ax_rank.set_title("Mutual Information 排名")
    ax_rank.grid(axis="x", alpha=0.2)

    im = ax_sel.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax_sel.set_xticks(np.arange(len(columns)))
    ax_sel.set_xticklabels([f"{method}\nλ={format_lambda(lam)}" for lam, method in columns], fontsize=10)
    ax_sel.tick_params(axis="y", left=False, labelleft=False)
    ax_sel.set_title("是否被選中")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if matrix[row, col]:
                ax_sel.text(col, row, "1", ha="center", va="center", fontsize=10, color="white")

    cbar = fig.colorbar(im, ax=ax_sel, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["未選", "選中"])

    fig.suptitle(f"圖二：MI 重要性與 GA/SA 選取狀態（seed {seed}）", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fig_dir / "02_feature_importance_selection.png", dpi=180)
    plt.close(fig)


def plot_pareto(run_dir: Path, fig_dir: Path, seed: int) -> None:
    metrics = pd.read_csv(run_dir / "metrics.csv", encoding="utf-8-sig")
    auc_col = "final_cv_auc"
    colors = {
        "GA": "#1f77b4",
        "SA": "#2ca02c",
        "RandomSearch": "#ff7f0e",
        "AllSearchFeatures": "#444444",
        "TopK": "#999999",
    }
    markers = {"GA": "o", "SA": "s", "RandomSearch": "^", "AllSearchFeatures": "D", "TopK": "x"}

    fig, ax = plt.subplots(figsize=(10, 7))
    for method in ["GA", "SA", "RandomSearch", "AllSearchFeatures"]:
        group = metrics[metrics["method"] == method].sort_values("lambda")
        if group.empty:
            continue
        label = "All Features" if method == "AllSearchFeatures" else method
        ax.scatter(
            group["selected_count"],
            group[auc_col],
            s=75,
            color=colors[method],
            marker=markers[method],
            label=label,
            alpha=0.92,
        )
        for _, row in group.iterrows():
            if method == "AllSearchFeatures" and row["lambda"] != 0:
                continue
            text = "All Features" if method == "AllSearchFeatures" else f"λ={format_lambda(row['lambda'])}"
            offset = (-86, 7) if method == "AllSearchFeatures" else (6, 5)
            ax.annotate(
                text,
                (row["selected_count"], row[auc_col]),
                textcoords="offset points",
                xytext=offset,
                fontsize=9,
                color=colors[method],
            )

    topk = metrics[metrics["method"].str.startswith("TopK_MI", na=False)]
    if not topk.empty:
        ax.scatter(
            topk["selected_count"],
            topk[auc_col],
            s=45,
            color=colors["TopK"],
            marker=markers["TopK"],
            label="Top-K MI",
            alpha=0.65,
        )

    ax.set_xlabel("選取特徵數")
    ax.set_ylabel("final 5-fold CV AUC")
    ax.set_title(f"圖三：Pareto Front - AUC vs 特徵數（seed {seed}）")
    ax.set_xlim(max(0, metrics["selected_count"].min() - 7), metrics["selected_count"].max() + 10)
    ax.set_ylim(metrics[auc_col].min() - 0.01, metrics[auc_col].max() + 0.008)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "03_pareto_auc_vs_features.png", dpi=180)
    plt.close(fig)


def plot_lambda_sensitivity(run_dir: Path, fig_dir: Path, seed: int) -> None:
    metrics = pd.read_csv(run_dir / "metrics.csv", encoding="utf-8-sig")
    methods = ["GA", "SA", "RandomSearch"]
    colors = {"GA": "#1f77b4", "SA": "#2ca02c", "RandomSearch": "#ff7f0e"}

    fig, (ax_auc, ax_feat) = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)
    for method in methods:
        group = metrics[metrics["method"] == method].sort_values("lambda")
        if group.empty:
            continue
        ax_auc.plot(
            group["lambda"],
            group["final_cv_auc"],
            marker="o",
            linewidth=2,
            color=colors[method],
            label=method,
        )
        ax_feat.plot(
            group["lambda"],
            group["selected_count"],
            marker="s",
            linewidth=2,
            color=colors[method],
            label=method,
        )

    all_row = metrics[(metrics["method"] == "AllSearchFeatures") & (np.isclose(metrics["lambda"], 0.0))]
    if not all_row.empty:
        all_row = all_row.iloc[0]
        ax_auc.axhline(
            all_row["final_cv_auc"],
            color="#555555",
            linestyle="--",
            linewidth=1.5,
            label="All Features AUC",
        )
        ax_feat.axhline(
            all_row["selected_count"],
            color="#555555",
            linestyle="--",
            linewidth=1.5,
            label="All Features count",
        )

    ax_auc.set_title("final CV AUC")
    ax_auc.set_xlabel("λ")
    ax_auc.set_ylabel("AUC")
    ax_auc.grid(alpha=0.25)
    ax_auc.legend(fontsize=9)

    ax_feat.set_title("選取特徵數")
    ax_feat.set_xlabel("λ")
    ax_feat.set_ylabel("特徵數")
    ax_feat.grid(alpha=0.25)
    ax_feat.legend(fontsize=9)

    fig.suptitle(f"圖四：λ 敏感度分析（seed {seed}）", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(fig_dir / "04_lambda_sensitivity.png", dpi=180)
    plt.close(fig)


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def update_html_images(aggregate_dir: Path, html_paths: list[Path], seed: int) -> None:
    fig_dir = aggregate_dir / "figures_for_report"
    blocks = []
    for filename, caption in zip(FIGURE_NAMES, FIGURE_CAPTIONS):
        image_b64 = encode_image(fig_dir / filename)
        blocks.append(
            f"<figure><img src='data:image/png;base64,{image_b64}' alt='{filename}' />"
            f"<figcaption>{caption}</figcaption></figure>"
        )

    note = (
        f"<p class=\"note\">以下四張圖皆為 seed {seed} 的代表性 run，用來讓報告流程更容易閱讀；"
        "multi-seed 穩定性與正式平均結論請以上方彙整表的 mean/std 為準。</p>"
    )
    figure_pattern = re.compile(
        r"<figure><img src='data:image/png;base64,[^']+'(?: alt='[^']*')?\s*/?>"
        r"<figcaption>.*?</figcaption></figure>",
        re.DOTALL,
    )
    note_pattern = re.compile(
        r"<p class=\"note\">以下四張圖皆為 .*?</p>",
        re.DOTALL,
    )

    for html_path in html_paths:
        text = html_path.read_text(encoding="utf-8")
        text, note_count = note_pattern.subn(note, text, count=1)
        if note_count == 0:
            text = text.replace("<h2>報告圖表</h2>", f"<h2>報告圖表</h2>\n{note}", 1)
        text, figure_count = figure_pattern.subn(lambda match, iterator=iter(blocks): next(iterator), text, count=4)
        if figure_count != 4:
            raise RuntimeError(f"Expected 4 figure blocks in {html_path}, replaced {figure_count}.")
        html_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregate-dir",
        type=Path,
        default=Path("results/aggregate_resume_20260615_135013"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--update-html", action="store_true")
    args = parser.parse_args()

    configure_matplotlib()
    aggregate_dir = args.aggregate_dir
    fig_dir = aggregate_dir / "figures_for_report"
    fig_dir.mkdir(parents=True, exist_ok=True)
    run_dir = find_seed_run(aggregate_dir, args.seed)

    plot_convergence(run_dir, fig_dir, args.seed)
    plot_feature_selection(run_dir, fig_dir, args.seed)
    plot_pareto(run_dir, fig_dir, args.seed)
    plot_lambda_sensitivity(run_dir, fig_dir, args.seed)

    if args.update_html:
        update_html_images(
            aggregate_dir,
            [
                aggregate_dir / "GA_SA_report_summary_zh_TW.html",
                Path("docs/index.html"),
            ],
            args.seed,
        )


if __name__ == "__main__":
    main()
