# GA/SA 特徵篩選期末專題

本專案比較 Genetic Algorithm（GA）與 Simulated Annealing（SA）在 IEEE-CIS Fraud Detection 高維資料上的特徵篩選效果。實驗依照課程 Word 要求設計，包含 proxy dataset 搜尋、完整訓練集 final 5-fold CV、multi-seed 穩定性分析，以及四張報告圖。

## 快速看結果

最建議先看這份中文 HTML 報告：

- GitHub Pages 入口：`https://musictw.github.io/GA_SA_feature_selection_project/`
- Repo 內 HTML：[`results/aggregate_resume_20260615_135013/GA_SA_report_summary_zh_TW.html`](results/aggregate_resume_20260615_135013/GA_SA_report_summary_zh_TW.html)
- Pages 用 HTML：[`docs/index.html`](docs/index.html)

如果 GitHub Pages 還沒開，先看 Repo 內 HTML；但 GitHub 只會顯示原始碼，真正要「直接用瀏覽器看報告」需要開 GitHub Pages。

## 目前結論摘要

正式彙整使用 `word_strict`，三個 seed：`7`, `42`, `2026`。

| lambda | 平均 AUC 最佳 | 平均 fitness 最佳 | 特徵數取捨重點 |
|---:|---|---|---|
| 0.00 | All Features, AUC 0.8572 | All Features, fitness 0.8572 | 不懲罰特徵數時，全特徵最高 |
| 0.05 | All Features, AUC 0.8572 | GA, fitness 0.8341 | GA 約 51 個特徵，AUC 接近全特徵 |
| 0.10 | All Features, AUC 0.8572 | GA, fitness 0.8217 | GA 約 38 個特徵，特徵數更少 |
| 0.20 | All Features, AUC 0.8572 | SA, fitness 0.8175 | SA 約 21 個特徵，最強稀疏化 |

重點解讀：All Features 的 AUC 最高，但使用 150 個特徵；GA/SA 的價值在於用更少特徵取得接近的 AUC，呈現 AUC 與特徵數量的 trade-off。

## Word 要求對照

本專案已包含：

- 使用 `train_transaction.csv`，不合併 identity
- GA/SA 搜尋使用 20,000 筆 stratified proxy rows
- 搜尋空間為 Mutual Information Top 150
- Fitness：`AUC - lambda * selected_features / total_features`
- lambda：`0`, `0.05`, `0.10`, `0.20`
- GA：population 40、generation 60、tournament k=3、crossover 0.85、mutation 1/150、elitism 2
- SA：1-3 bit flip、calibration 100 moves、alpha 0.98、T_min 1e-4、iterations 5000
- Baselines：Random Search、All Features、Top-K MI
- final evaluation：完整 `590,540` 筆 train_transaction + 5-fold CV
- multi-seed mean/std、計算時間、GA/SA Jaccard overlap
- Word 要求四張圖：收斂曲線、特徵重要性、Pareto Front、lambda sensitivity

注意：目前彙整中 Random Search 的 trial 數有一個小差異，seed 7 是 2400 次，seed 42/2026 是 5000 次；GA/SA 主設定一致。HTML 報告中也有標註這個風險。

## 主要檔案

| 路徑 | 用途 |
|---|---|
| `GA_SA_feature_selection_best_zh_TW.ipynb` | 最完整中文 all-in-one notebook |
| `GA_SA_feature_selection_best_fast_zh_TW.ipynb` | 加速重跑版本 |
| `src/ml_feature_selection/` | 模組化 Python 實作 |
| `run_experiment.py` | 單次實驗入口 |
| `run_multi_seed.py` | 多 seed 彙整入口 |
| `resume_word_strict_remaining_seeds.py` | 補跑 strict seeds 用 |
| `results/aggregate_resume_20260615_135013/` | 最終彙整結果 |
| `docs/index.html` | GitHub Pages 直接瀏覽入口 |

## 如何重跑

本 repo 不包含 Kaggle 原始資料與 cache。請自行下載 IEEE-CIS Fraud Detection，並放入：

```text
data/raw/train_transaction.csv
```

先跑 smoke test：

```powershell
python run_experiment.py --mode synthetic --preset smoke
```

若要用 notebook 跑，打開：

```text
GA_SA_feature_selection_best_zh_TW.ipynb
```

若只想較快驗證流程，打開：

```text
GA_SA_feature_selection_best_fast_zh_TW.ipynb
```

## GitHub Pages 開啟方式

若要讓別人用網址直接看結果，需要：

1. 到 GitHub repo 的 `Settings`
2. 左側選 `Pages`
3. Source 選 `Deploy from a branch`
4. Branch 選 `main`
5. Folder 選 `/docs`
6. Save

開啟後網址會是：

```text
https://musictw.github.io/GA_SA_feature_selection_project/
```

如果 repo 維持 private，別人通常看不到 repo 內容；要給老師或同學看，需要把 repo 改 public，或把對方加成 collaborator。
