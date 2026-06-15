# GA/SA 特徵篩選專案使用說明

這個資料夾是「非 all-in-one」版本的完整專案包。它保留 Python 檔案分工，不把所有程式塞進一個 notebook，因此比較容易維護，也比較不容易遇到 notebook 原始 JSON 編碼問題。

## 最新交付版本

目前最完整的交付結果在：

```text
results/aggregate_resume_20260615_135013/GA_SA_report_summary_zh_TW.html
```

這份 HTML 已包含 Word 要求的四張圖、三個 seed 的 `word_strict` 彙整、完整資料集 final 5-fold CV 結果，以及中文完整性檢查。

建議主要使用的 notebook 是：

```text
GA_SA_feature_selection_best_zh_TW.ipynb
```

若只想較快重跑流程，可用：

```text
GA_SA_feature_selection_best_fast_zh_TW.ipynb
```

GitHub 版本不包含 Kaggle 原始 CSV 與 `data/cache/`，請自行把 `train_transaction.csv` 放入 `data/raw/` 後再重跑。

## 這包裡有什麼

```text
GA_SA_feature_selection_project/
  GA_SA_feature_selection.ipynb
  GA_SA_feature_selection_best_zh_TW.ipynb
  GA_SA_feature_selection_best_fast_zh_TW.ipynb
  run_experiment.py
  run_multi_seed.py
  aggregate_results.py
  run_smoke.ps1
  run_quick_real.ps1
  run_multi_seed_quick.ps1
  requirements.txt
  README_zh_TW.md
  PYTHON流程說明.md
  src/ml_feature_selection/
  data/raw/
  data/cache/
  results/aggregate_resume_20260615_135013/
```

重點檔案：

- `GA_SA_feature_selection_best_zh_TW.ipynb`：目前最完整的中文 all-in-one notebook。
- `GA_SA_feature_selection_best_fast_zh_TW.ipynb`：較快重跑用的中文 notebook。
- `GA_SA_feature_selection.ipynb`：模組化 notebook 入口，會呼叫這包裡的 `.py` 檔。
- `run_experiment.py`：單次實驗主程式。
- `run_multi_seed.py`：多 seed 批次實驗。
- `aggregate_results.py`：彙總多個 `results/run_*`。
- `PYTHON流程說明.md`：更細的程式流程解釋。
- `results/aggregate_resume_20260615_135013/`：已完成的 Word strict 彙整結果與中文 HTML 報告。

## 先放資料

因為 Kaggle CSV 很大，這個 package 沒有複製原始資料。請把：

```text
train_transaction.csv
```

放到：

```text
GA_SA_feature_selection_project/data/raw/train_transaction.csv
```

目前程式沒有合併 `train_identity.csv`，只使用 `train_transaction.csv`。

## 建議執行順序

先進入 package 資料夾：

```powershell
cd GA_SA_feature_selection_project
```

### 1. 先跑 smoke 測試

這不需要 Kaggle 資料，用 synthetic data 確認整套流程能跑。

```powershell
powershell -ExecutionPolicy Bypass -File .\run_smoke.ps1
```

如果成功，會產生：

```text
results/run_時間/
```

### 2. 跑正式 quick

確認 `data/raw/train_transaction.csv` 已經放好後執行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_quick_real.ps1
```

`quick` 是筆電優先設定：

```text
proxy_rows = 20000
search_features = 80
lambda = 0.05, 0.10
GA generations = 18
SA iterations = 600
final CV = 3-fold
```

### 3. 跑多 seed

如果要讓報告更有說服力，跑：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_multi_seed_quick.ps1
```

它會跑：

```text
seed = 42, 7, 13
preset = quick
```

跑完會多一個：

```text
results/aggregate_時間/
```

裡面有平均與標準差。

## 用 Notebook 跑

打開：

```text
GA_SA_feature_selection.ipynb
```

Notebook 預設只跑 smoke，不會自動跑正式資料。

要跑正式資料時，找到正式資料那格，改成：

```python
RUN_REAL = True
```

要跑多 seed 時，找到多 seed 那格，改成：

```python
RUN_MULTI_SEED = True
```

如果只是想看目前結果，直接跑「查看最新結果」和「查看多 Seed 彙總結果」的 cell 即可。

## 輸出結果怎麼看

每次實驗會輸出到：

```text
results/run_YYYYMMDD_HHMMSS/
```

主要檔案：

```text
summary.md
metrics.csv
selected_features.json
convergence.csv
plots/convergence.png
plots/pareto_auc_vs_features.png
plots/lambda_sensitivity.png
run_metadata.json
```

最重要的是 `metrics.csv` 裡的：

```text
final_cv_auc
final_cv_fitness
selected_count
```

不要只看 inner-loop 的 `auc`，因為 GA/SA 在搜尋階段可能對 holdout validation 有一點 overfitting。

## 已附的正式結果

這包裡附了一份已跑完的結果：

```text
results/aggregate_resume_20260615_135013/
```

它的設定是：

```text
preset = word_strict
seeds = 7, 42, 2026
proxy rows = 20000
search_features = 150
lambda = 0.0, 0.05, 0.10, 0.20
final rows = 590540
final CV = 5-fold on full train_transaction.csv
```

主要輸出：

```text
GA_SA_report_summary_zh_TW.html
aggregate_metrics.csv
all_metrics.csv
feature_overlap_summary.csv
figures_for_report/
```

注意：

```text
All Features 的平均 AUC 最高，但使用 150 個特徵；GA/SA 呈現的是 AUC 與特徵數量之間的取捨。
```

因為 AUC 差距不大。

## Cache 說明

正式資料前處理會自動 cache 到：

```text
data/cache/
```

第一次跑會比較久，因為要讀大 CSV、補缺失值、類別編碼、Variance Threshold、MI 預篩。

之後如果資料與參數相同，就會直接讀 cache。

如果要強制重建：

```powershell
python run_experiment.py --mode real --data-path .\data\raw\train_transaction.csv --preset quick --no-cache
```

## 編碼注意

這個資料夾的中文說明檔是 UTF-8。建議用 VS Code、Jupyter、Notepad、Typora 或 Obsidian 開啟。

如果 PowerShell 顯示中文路徑亂碼，請用這些 `.ps1` 腳本執行，因為它們已經設定：

```text
PYTHONUTF8=1
PYTHONIOENCODING=utf-8
chcp 65001
```

## 建議交作業流程

1. 先用 `results/run_20260611_153207/` 寫初版結果。
2. 如果時間夠，跑 `run_multi_seed_quick.ps1`。
3. 用 `aggregate_summary.md` 補穩定性分析。
4. 報告中引用 final CV，不引用 inner AUC 當最終結論。
