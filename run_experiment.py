from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml_feature_selection.config import PRESETS, get_config
from ml_feature_selection.data import (
    load_ieee_transaction,
    load_synthetic,
    prepare_ieee_transaction_with_cache,
    preprocess_and_select,
)
from ml_feature_selection.evaluation import FitnessEvaluator
from ml_feature_selection.optimizers import run_ga, run_random_search, run_sa
from ml_feature_selection.reporting import (
    make_run_dir,
    selected_feature_names,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GA/SA feature-selection experiment.")
    parser.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--data-path", type=Path, default=ROOT / "data" / "raw" / "train_transaction.csv")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache")
    parser.add_argument("--no-cache", action="store_true", help="Disable real-data preprocessing cache.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--proxy-rows", type=int, default=None)
    parser.add_argument("--search-features", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config(
        args.preset,
        proxy_rows=args.proxy_rows,
        search_features=args.search_features,
        random_seed=args.seed,
    )
    run_dir = make_run_dir(args.results_dir)

    print(f"[1/5] Loading data: mode={args.mode}, preset={config.preset}", flush=True)
    if args.mode == "synthetic":
        X_raw, y = load_synthetic(config.proxy_rows, config.random_seed)
        source = "synthetic"
        print("[2/5] Preprocessing and selecting MI search space", flush=True)
        data = preprocess_and_select(
            X_raw=X_raw,
            y=y,
            search_features=config.search_features,
            random_seed=config.random_seed,
            source=source,
        )
    elif args.no_cache:
        X_raw, y = load_ieee_transaction(args.data_path, config.proxy_rows, config.random_seed)
        source = str(args.data_path)
        print("[2/5] Preprocessing and selecting MI search space (cache disabled)", flush=True)
        data = preprocess_and_select(
            X_raw=X_raw,
            y=y,
            search_features=config.search_features,
            random_seed=config.random_seed,
            source=source,
        )
    else:
        print("[2/5] Checking preprocessing cache", flush=True)
        data_start = perf_counter()
        data, cache_hit, cache_path = prepare_ieee_transaction_with_cache(
            path=args.data_path,
            proxy_rows=config.proxy_rows,
            search_features=config.search_features,
            random_seed=config.random_seed,
            cache_dir=args.cache_dir,
        )
        state = "hit" if cache_hit else "miss/rebuilt"
        print(
            f"      cache {state}: {cache_path} ({perf_counter() - data_start:.1f}s)",
            flush=True,
        )
    print(
        f"      rows={data.rows}, original_features={data.original_features}, "
        f"search_features={len(data.feature_names)}",
        flush=True,
    )

    metrics: list[dict[str, object]] = []
    selected: dict[str, list[str]] = {}
    convergence: list[dict[str, object]] = []
    total_start = perf_counter()

    for lam_idx, penalty_lambda in enumerate(config.lambdas, start=1):
        print(f"[3/5] Lambda {lam_idx}/{len(config.lambdas)}: {penalty_lambda}", flush=True)
        rng = np.random.default_rng(config.random_seed + int(penalty_lambda * 10_000))
        evaluator = FitnessEvaluator(
            X=data.X,
            y=data.y,
            penalty_lambda=penalty_lambda,
            cv_folds=config.inner_cv_folds,
            test_size=config.test_size,
            n_estimators=config.xgb_estimators_inner,
            max_depth=config.xgb_max_depth,
            learning_rate=config.xgb_learning_rate,
            device=args.device,
            random_seed=config.random_seed,
        )

        all_mask = np.ones(data.X.shape[1], dtype=bool)
        all_result = evaluator.evaluate(all_mask)
        _add_metric(
            metrics,
            selected,
            data.feature_names,
            "AllSearchFeatures",
            penalty_lambda,
            all_mask,
            all_result,
            all_result.seconds,
            evaluator.model_fits,
        )

        random_result = run_random_search(
            evaluator=evaluator,
            trials=config.random_trials,
            rng=rng,
            penalty_lambda=penalty_lambda,
            progress_callback=_progress,
        )
        _record_search(random_result, metrics, selected, convergence, data.feature_names, evaluator.model_fits)
        print(
            f"      RandomSearch: auc={random_result.result.auc:.4f}, "
            f"features={random_result.result.selected_count}, seconds={random_result.seconds:.1f}",
            flush=True,
        )

        mutation_rate = config.ga_mutation_rate or (1.0 / data.X.shape[1])
        ga_result = run_ga(
            evaluator=evaluator,
            population_size=config.ga_population,
            generations=config.ga_generations,
            elites=config.ga_elites,
            tournament_k=config.ga_tournament_k,
            crossover_rate=config.ga_crossover_rate,
            mutation_rate=mutation_rate,
            rng=rng,
            penalty_lambda=penalty_lambda,
            progress_callback=_progress,
        )
        _record_search(ga_result, metrics, selected, convergence, data.feature_names, evaluator.model_fits)
        print(
            f"      GA: auc={ga_result.result.auc:.4f}, "
            f"features={ga_result.result.selected_count}, seconds={ga_result.seconds:.1f}",
            flush=True,
        )

        sa_result = run_sa(
            evaluator=evaluator,
            iterations=config.sa_iterations,
            alpha=config.sa_alpha,
            t_min=config.sa_t_min,
            calibration_moves=config.sa_calibration_moves,
            log_interval=config.sa_log_interval,
            rng=rng,
            penalty_lambda=penalty_lambda,
            progress_callback=_progress,
        )
        _record_search(sa_result, metrics, selected, convergence, data.feature_names, evaluator.model_fits)
        print(
            f"      SA: auc={sa_result.result.auc:.4f}, "
            f"features={sa_result.result.selected_count}, seconds={sa_result.seconds:.1f}",
            flush=True,
        )

        topk_counts = sorted({ga_result.result.selected_count, sa_result.result.selected_count})
        for k in topk_counts:
            if k <= 0:
                continue
            topk_mask = np.zeros(data.X.shape[1], dtype=bool)
            topk_mask[: min(k, data.X.shape[1])] = True
            topk_result = evaluator.evaluate(topk_mask)
            _add_metric(
                metrics,
                selected,
                data.feature_names,
                f"TopK_MI_{k}",
                penalty_lambda,
                topk_mask,
                topk_result,
                topk_result.seconds,
                evaluator.model_fits,
            )

    print("[4/5] Final cross-validation re-evaluation", flush=True)
    _final_cv_reevaluate(metrics, data, config, args.device)

    print("[5/5] Writing outputs", flush=True)
    write_outputs(
        run_dir,
        config,
        data,
        metrics,
        selected,
        convergence,
        extra_metadata={
            "mode": args.mode,
            "device": args.device,
            "data_path": str(args.data_path),
            "cache_dir": str(args.cache_dir),
            "no_cache": bool(args.no_cache),
        },
    )
    print(f"Done in {perf_counter() - total_start:.1f}s", flush=True)
    print(f"Results: {run_dir}", flush=True)


def _record_search(
    search_result,
    metrics: list[dict[str, object]],
    selected: dict[str, list[str]],
    convergence: list[dict[str, object]],
    feature_names: list[str],
    model_fits: int,
) -> None:
    _add_metric(
        metrics,
        selected,
        feature_names,
        search_result.method,
        search_result.penalty_lambda,
        search_result.mask,
        search_result.result,
        search_result.seconds,
        model_fits,
    )
    convergence.extend(search_result.history)


def _progress(message: str) -> None:
    print(f"      {message}", flush=True)


def _add_metric(
    metrics: list[dict[str, object]],
    selected: dict[str, list[str]],
    feature_names: list[str],
    method: str,
    penalty_lambda: float,
    mask: np.ndarray,
    result,
    seconds: float,
    model_fits: int,
) -> None:
    key = f"lambda={penalty_lambda}::{method}"
    selected[key] = selected_feature_names(feature_names, mask)
    metrics.append(
        {
            "method": method,
            "lambda": penalty_lambda,
            "mask_bits": "".join("1" if value else "0" for value in mask.astype(bool)),
            "fitness": result.fitness,
            "auc": result.auc,
            "selected_count": result.selected_count,
            "selected_ratio": result.selected_ratio,
            "seconds": seconds,
            "model_fits_so_far": model_fits,
        }
    )


def _final_cv_reevaluate(metrics: list[dict[str, object]], data, config, device: str) -> None:
    evaluators: dict[float, FitnessEvaluator] = {}
    total = len(metrics)
    for idx, row in enumerate(metrics, start=1):
        penalty_lambda = float(row["lambda"])
        if penalty_lambda not in evaluators:
            evaluators[penalty_lambda] = FitnessEvaluator(
                X=data.X,
                y=data.y,
                penalty_lambda=penalty_lambda,
                cv_folds=config.final_cv_folds,
                test_size=config.test_size,
                n_estimators=config.xgb_estimators_final,
                max_depth=config.xgb_max_depth,
                learning_rate=config.xgb_learning_rate,
                device=device,
                random_seed=config.random_seed + 7_777,
            )
        mask = np.array([char == "1" for char in str(row["mask_bits"])], dtype=bool)
        print(
            f"      Final CV {idx}/{total}: {row['method']} lambda={penalty_lambda}",
            flush=True,
        )
        result = evaluators[penalty_lambda].evaluate(mask)
        row["final_cv_auc"] = result.auc
        row["final_cv_fitness"] = result.fitness
        row["final_cv_seconds"] = result.seconds
        row["final_cv_folds"] = config.final_cv_folds


if __name__ == "__main__":
    main()
