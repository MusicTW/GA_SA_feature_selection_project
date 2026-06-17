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
    "收斂曲線比較：3 個 seed 的平均曲線；陰影代表跨 seed 的 ±1 標準差。x 軸正規化為搜尋進度百分比，讓 GA 60 generations 與 SA 5000 iterations 可在同一張圖比較。",
    "特徵重要性與選取頻率：左側為 Mutual Information 前 30 名特徵；右側色塊為 3 個 seed 中被 GA/SA 選中的次數，數字越高代表越穩定被選到。",
    "Pareto Front：3-seed 平均 final CV AUC 對平均選取特徵數；誤差棒代表跨 seed 的 ±1 標準差，用來呈現 AUC 與特徵數的取捨。",
    "Lambda 敏感度分析：3-seed 平均 AUC 與平均選取特徵數；誤差棒代表跨 seed 的 ±1 標準差，用來觀察 λ 增加後的 trade-off。",
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


def load_runs(aggregate_dir: Path) -> list[dict[str, object]]:
    runs = pd.read_csv(aggregate_dir / "runs.csv", encoding="utf-8-sig")
    result: list[dict[str, object]] = []
    for row in runs.to_dict("records"):
        run_dir = Path(str(row["run_dir"]))
        if not run_dir.is_absolute():
            run_dir = Path.cwd() / run_dir
        result.append({"seed": int(row["seed"]), "run_dir": run_dir})
    return result


def format_lambda(value: float) -> str:
    if abs(value) < 1e-12:
        return "0"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def selected_key(lam: float, method: str) -> str:
    text = str(lam)
    return f"lambda={text}::{method}"


def plot_convergence(runs: list[dict[str, object]], fig_dir: Path) -> None:
    grid = np.linspace(0.0, 100.0, 101)
    lambdas = [0.0, 0.05, 0.1, 0.2]
    series_specs = [
        ("GA", "best_fitness", "GA best fitness", "#1f77b4", "-"),
        ("GA", "avg_or_current_fitness", "GA average fitness", "#ff7f0e", "--"),
        ("SA", "best_fitness", "SA best fitness", "#2ca02c", "-"),
    ]
    values: dict[tuple[float, str, str], list[np.ndarray]] = {}

    for run in runs:
        conv = pd.read_csv(Path(run["run_dir"]) / "convergence.csv", encoding="utf-8-sig")
        for lam in lambdas:
            for method, column, _, _, _ in series_specs:
                group = conv[(conv["method"] == method) & (np.isclose(conv["lambda"], lam))]
                if group.empty or column not in group.columns:
                    continue
                group = group.sort_values("step")
                max_step = float(group["step"].max())
                progress = group["step"].astype(float).to_numpy() / max_step * 100.0
                y = group[column].astype(float).to_numpy()
                values.setdefault((lam, method, column), []).append(
                    np.interp(grid, progress, y, left=y[0], right=y[-1])
                )

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    axes = axes.ravel()
    handles = []
    labels = []
    for ax, lam in zip(axes, lambdas):
        for method, column, label, color, linestyle in series_specs:
            arrs = values.get((lam, method, column), [])
            if not arrs:
                continue
            stacked = np.vstack(arrs)
            mean = stacked.mean(axis=0)
            std = stacked.std(axis=0, ddof=1) if stacked.shape[0] > 1 else np.zeros_like(mean)
            (line,) = ax.plot(grid, mean, label=label, color=color, linestyle=linestyle, linewidth=2.0)
            ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.13, linewidth=0)
            if label not in labels:
                handles.append(line)
                labels.append(label)
        ax.set_title(f"λ = {format_lambda(lam)}", fontsize=13)
        ax.set_xlabel("搜尋進度 (%)")
        ax.set_ylabel("Fitness")
        ax.grid(alpha=0.25)

    fig.suptitle("圖一：GA/SA 收斂曲線比較（3-seed mean ± std）", fontsize=17)
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.text(
        0.01,
        0.01,
        "說明：每條線為 seed 7、42、2026 的平均；陰影為 ±1 標準差。GA 為 60 generations，SA 為 5000 iterations。",
        fontsize=10,
        color="#3e4c59",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    fig.savefig(fig_dir / "01_convergence.png", dpi=180)
    plt.close(fig)


def load_feature_order(runs: list[dict[str, object]]) -> list[str]:
    first = Path(runs[0]["run_dir"]) / "selected_features.json"
    selected = json.loads(first.read_text(encoding="utf-8"))
    values = selected.get("lambda=0.0::AllSearchFeatures")
    if not values:
        raise RuntimeError("Cannot find lambda=0.0::AllSearchFeatures for MI feature order.")
    return list(values)


def plot_feature_frequency(runs: list[dict[str, object]], fig_dir: Path, top_n: int = 30) -> None:
    feature_order = load_feature_order(runs)
    top_features = feature_order[:top_n]
    columns = [(0.05, "GA"), (0.05, "SA"), (0.1, "GA"), (0.1, "SA")]
    matrix = np.zeros((top_n, len(columns)), dtype=int)

    for run in runs:
        selected = json.loads((Path(run["run_dir"]) / "selected_features.json").read_text(encoding="utf-8"))
        for col_idx, (lam, method) in enumerate(columns):
            feature_set = set(selected.get(selected_key(lam, method), []))
            for row_idx, feature in enumerate(top_features):
                if feature in feature_set:
                    matrix[row_idx, col_idx] += 1

    rank_score = np.arange(top_n, 0, -1)
    labels = [f"{i + 1:02d}. {feature}" for i, feature in enumerate(top_features)]
    y = np.arange(top_n)

    fig, (ax_rank, ax_freq) = plt.subplots(
        ncols=2,
        figsize=(14, 10),
        gridspec_kw={"width_ratios": [1.4, 1.2]},
        sharey=True,
    )
    ax_rank.barh(y, rank_score, color="#4c78a8", alpha=0.88)
    ax_rank.set_yticks(y)
    ax_rank.set_yticklabels(labels, fontsize=9)
    ax_rank.invert_yaxis()
    ax_rank.set_xlabel("MI rank score")
    ax_rank.set_title("Mutual Information 排名")
    ax_rank.grid(axis="x", alpha=0.2)

    im = ax_freq.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=len(runs))
    ax_freq.set_xticks(np.arange(len(columns)))
    ax_freq.set_xticklabels([f"{method}\nλ={format_lambda(lam)}" for lam, method in columns], fontsize=10)
    ax_freq.tick_params(axis="y", left=False, labelleft=False)
    ax_freq.set_title("3 個 seed 的選取次數")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            color = "white" if value >= 2 else "#111111"
            ax_freq.text(col, row, str(value), ha="center", va="center", fontsize=9, color=color)

    cbar = fig.colorbar(im, ax=ax_freq, fraction=0.046, pad=0.04)
    cbar.set_label("被選中次數（0-3）")
    cbar.set_ticks([0, 1, 2, 3])

    fig.suptitle("圖二：MI 重要性與 GA/SA 選取頻率（3-seed aggregate）", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fig_dir / "02_feature_importance_selection.png", dpi=180)
    plt.close(fig)


def plot_pareto(aggregate_dir: Path, fig_dir: Path) -> None:
    summary = pd.read_csv(aggregate_dir / "aggregate_metrics.csv", encoding="utf-8-sig")
    methods = ["GA", "SA", "RandomSearch"]
    colors = {"GA": "#1f77b4", "SA": "#2ca02c", "RandomSearch": "#ff7f0e", "AllSearchFeatures": "#444444"}
    markers = {"GA": "o", "SA": "s", "RandomSearch": "^", "AllSearchFeatures": "D"}

    fig, ax = plt.subplots(figsize=(10, 7))
    for method in methods:
        group = summary[summary["method"] == method].sort_values("lambda")
        ax.errorbar(
            group["selected_count_mean"],
            group["final_cv_auc_mean"],
            xerr=group["selected_count_std"],
            yerr=group["final_cv_auc_std"],
            fmt=markers[method],
            color=colors[method],
            capsize=3,
            markersize=7,
            linestyle="none",
            label=method,
            alpha=0.92,
        )
        for _, row in group.iterrows():
            ax.annotate(
                f"λ={format_lambda(row['lambda'])}",
                (row["selected_count_mean"], row["final_cv_auc_mean"]),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=9,
                color=colors[method],
            )

    all_row = summary[summary["method"] == "AllSearchFeatures"].sort_values("lambda").iloc[0]
    ax.errorbar(
        [all_row["selected_count_mean"]],
        [all_row["final_cv_auc_mean"]],
        xerr=[all_row["selected_count_std"]],
        yerr=[all_row["final_cv_auc_std"]],
        fmt=markers["AllSearchFeatures"],
        color=colors["AllSearchFeatures"],
        capsize=3,
        markersize=8,
        linestyle="none",
        label="All Features",
    )
    ax.annotate(
        "All Features\n150 features",
        (all_row["selected_count_mean"], all_row["final_cv_auc_mean"]),
        textcoords="offset points",
        xytext=(-78, -28),
        fontsize=9,
        color=colors["AllSearchFeatures"],
    )
    ax.set_xlabel("平均選取特徵數")
    ax.set_ylabel("平均 final 5-fold CV AUC")
    ax.set_title("圖三：Pareto Front - AUC vs 特徵數（3-seed mean ± std）")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "03_pareto_auc_vs_features.png", dpi=180)
    plt.close(fig)


def plot_lambda_sensitivity(aggregate_dir: Path, fig_dir: Path) -> None:
    summary = pd.read_csv(aggregate_dir / "aggregate_metrics.csv", encoding="utf-8-sig")
    methods = ["GA", "SA", "RandomSearch"]
    colors = {"GA": "#1f77b4", "SA": "#2ca02c", "RandomSearch": "#ff7f0e"}

    fig, (ax_auc, ax_feat) = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)
    for method in methods:
        group = summary[summary["method"] == method].sort_values("lambda")
        ax_auc.errorbar(
            group["lambda"],
            group["final_cv_auc_mean"],
            yerr=group["final_cv_auc_std"],
            marker="o",
            capsize=3,
            linewidth=2,
            color=colors[method],
            label=method,
        )
        ax_feat.errorbar(
            group["lambda"],
            group["selected_count_mean"],
            yerr=group["selected_count_std"],
            marker="s",
            capsize=3,
            linewidth=2,
            color=colors[method],
            label=method,
        )

    all_row = summary[summary["method"] == "AllSearchFeatures"].sort_values("lambda").iloc[0]
    ax_auc.axhline(
        all_row["final_cv_auc_mean"],
        color="#555555",
        linestyle="--",
        linewidth=1.5,
        label="All Features AUC",
    )
    ax_feat.axhline(
        all_row["selected_count_mean"],
        color="#555555",
        linestyle="--",
        linewidth=1.5,
        label="All Features count",
    )

    ax_auc.set_title("平均 final CV AUC")
    ax_auc.set_xlabel("λ")
    ax_auc.set_ylabel("AUC")
    ax_auc.grid(alpha=0.25)
    ax_auc.legend(fontsize=9)

    ax_feat.set_title("平均選取特徵數")
    ax_feat.set_xlabel("λ")
    ax_feat.set_ylabel("特徵數")
    ax_feat.grid(alpha=0.25)
    ax_feat.legend(fontsize=9)

    fig.suptitle("圖四：λ 敏感度分析（3-seed mean ± std）", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(fig_dir / "04_lambda_sensitivity.png", dpi=180)
    plt.close(fig)


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def update_html_images(aggregate_dir: Path, html_paths: list[Path]) -> None:
    fig_dir = aggregate_dir / "figures_for_report"
    blocks = []
    for filename, caption in zip(FIGURE_NAMES, FIGURE_CAPTIONS):
        image_b64 = encode_image(fig_dir / filename)
        blocks.append(
            f"<figure><img src='data:image/png;base64,{image_b64}' alt='{filename}' />"
            f"<figcaption>{caption}</figcaption></figure>"
        )

    note = (
        "<p class=\"note\">以下四張圖皆為 3-seed 彙整圖（seed 7、42、2026），"
        "不是單次 seed 2026；陰影或誤差棒代表跨 seed 的標準差。</p>"
    )
    pattern = re.compile(
        r"<figure><img src='data:image/png;base64,[^']+'(?: alt='[^']*')?\s*/?>"
        r"<figcaption>.*?</figcaption></figure>",
        re.DOTALL,
    )

    for html_path in html_paths:
        text = html_path.read_text(encoding="utf-8")
        text, count = pattern.subn(lambda match, iterator=iter(blocks): next(iterator), text, count=4)
        if count != 4:
            raise RuntimeError(f"Expected 4 figure blocks in {html_path}, replaced {count}.")
        if "以下四張圖皆為 3-seed 彙整圖" not in text:
            text = text.replace("<h2>報告圖表</h2>", f"<h2>報告圖表</h2>\n{note}", 1)
        html_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregate-dir",
        type=Path,
        default=Path("results/aggregate_resume_20260615_135013"),
    )
    parser.add_argument("--update-html", action="store_true")
    args = parser.parse_args()

    configure_matplotlib()
    aggregate_dir = args.aggregate_dir
    fig_dir = aggregate_dir / "figures_for_report"
    fig_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(aggregate_dir)
    plot_convergence(runs, fig_dir)
    plot_feature_frequency(runs, fig_dir)
    plot_pareto(aggregate_dir, fig_dir)
    plot_lambda_sensitivity(aggregate_dir, fig_dir)

    if args.update_html:
        update_html_images(
            aggregate_dir,
            [
                aggregate_dir / "GA_SA_report_summary_zh_TW.html",
                Path("docs/index.html"),
            ],
        )


if __name__ == "__main__":
    main()
