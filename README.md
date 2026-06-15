# GA vs SA Feature Selection

This project implements the final-project experiment described in the Word file:
compare Genetic Algorithm (GA) and Simulated Annealing (SA) for high-dimensional
feature selection on the IEEE-CIS Fraud Detection dataset.

## Final Report

The current report-ready output is:

```text
results/aggregate_resume_20260615_135013/GA_SA_report_summary_zh_TW.html
```

It includes the Word-required figures, three `word_strict` seeds, full
`train_transaction.csv` final 5-fold CV evaluation, aggregate metrics, and
feature-overlap summaries.

Kaggle raw CSV files and preprocessing caches are intentionally excluded from
Git. After cloning, put `train_transaction.csv` under `data/raw/` before
running real-data experiments.

## Quick Run

Run the complete pipeline on synthetic data first:

```powershell
python run_experiment.py --mode synthetic --preset smoke
```

or:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_smoke.ps1
```

The output is written to `results\run_<timestamp>\`:

- `summary.md`: Chinese experiment summary
- `metrics.csv`: AUC, fitness, selected feature counts, runtime
- `selected_features.json`: selected feature names
- `convergence.csv`: GA/SA convergence trace
- `plots\*.png`: convergence, Pareto, and lambda plots

Aggregate outputs from multi-seed runs are written to `results\aggregate_<timestamp>\`:

- `aggregate_summary.md`: best method per lambda by mean AUC and mean fitness
- `aggregate_metrics.csv`: mean/std metrics grouped by lambda and method
- `all_metrics.csv`: raw metric rows from every run
- `runs.csv`: run directories and metadata

For real-data runs, preprocessing is cached under `data\cache\`. Re-running the
same CSV with the same `proxy_rows`, `search_features`, and seed will skip the
slow CSV read / encoding / MI-selection step. Use `--no-cache` only when you
want to force rebuilding preprocessing.

## Run With Kaggle Data

Download Kaggle IEEE-CIS Fraud Detection and put this file here:

```text
data\raw\train_transaction.csv
```

Then run:

```powershell
python run_experiment.py --mode real --data-path .\data\raw\train_transaction.csv --preset quick --device cpu
```

or:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_quick_real.ps1
```

For a more report-ready stability check, run multiple seeds and aggregate them:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_multi_seed_quick.ps1
```

This runs seeds `42, 7, 13` with the `quick` preset and writes one aggregate
folder under `results\`.

## Presets

`smoke` is for checking the full pipeline quickly.

`quick` is the recommended first real-data run on this laptop. It uses a proxy
dataset, a reduced search budget, and holdout validation inside GA/SA.

`standard` is slower and closer to the report plan. Use it after `quick` works.

## Notes

- The default device is CPU. On this laptop, small XGBoost jobs are often faster
  on CPU than on the RTX 3050 GPU because the dataset slices are small.
- The GA/SA inner loop uses holdout validation by default for speed. The final
  selected feature subsets are re-evaluated with cross-validation.
- Fitness is:

```text
fitness = AUC - lambda * (selected_features / total_search_features)
```

- SA uses the correct maximization acceptance rule:

```text
accept worse solution with probability exp((new_fitness - old_fitness) / T)
```
