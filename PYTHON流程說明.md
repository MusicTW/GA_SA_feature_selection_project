# Python 實驗流程說明

這份文件說明 `D:\ML期末` 內 Python 程式如何完成「GA 與 SA 用於特徵篩選」實驗。它對應目前實作，不是理論版流程。

## 一、主要執行入口

主要入口是：

```powershell
python run_experiment.py --mode real --data-path .\data\raw\train_transaction.csv --preset quick --device cpu
```

也可以用 PowerShell 腳本：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_quick_real.ps1
```

Notebook 入口是：

```text
GA_SA_feature_selection.ipynb
```

Notebook 內正式資料預設不會自動長跑。要跑正式資料時，需要把：

```python
RUN_REAL = False
```

改成：

```python
RUN_REAL = True
```

如果要做多 seed 穩定性比較，可以跑：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_multi_seed_quick.ps1
```

這會連續跑 `seed = 42, 7, 13` 的 quick preset，最後自動產生一個 `results/aggregate_*` 彙總資料夾。

## 二、專案檔案分工

```text
run_experiment.py
```

總控制程式。負責讀參數、載入資料、跑 baseline、跑 GA、跑 SA、做 final CV，最後輸出結果。

```text
src/ml_feature_selection/config.py
```

定義 `smoke`、`quick`、`standard` 三種實驗規模。

```text
src/ml_feature_selection/data.py
```

負責資料載入、前處理、MI 預篩，以及正式資料的 preprocessing cache。

```text
src/ml_feature_selection/evaluation.py
```

負責計算某一組特徵 mask 的 AUC 與 fitness。

```text
src/ml_feature_selection/optimizers.py
```

實作 Random Search、GA、SA。

```text
src/ml_feature_selection/reporting.py
```

負責輸出 `metrics.csv`、`selected_features.json`、`convergence.csv`、`summary.md` 和圖表。

```text
src/ml_feature_selection/aggregation.py
```

負責把多個 `results/run_*` 的結果彙整成平均與標準差。

```text
run_multi_seed.py
```

批次執行多個 seed，跑完後自動呼叫彙總器。

```text
aggregate_results.py
```

只做彙總，不重新跑實驗。適合拿既有的多個 `run_*` 結果重新整理。

## 三、整體資料流

程式流程可以看成：

```text
train_transaction.csv
  -> 讀取資料
  -> stratified sampling 取 20,000 筆
  -> 缺失值處理與類別編碼
  -> Variance Threshold 移除零變異特徵
  -> Mutual Information 預篩前 80 個特徵
  -> 建立 X, y, feature_names
  -> baseline / GA / SA 共用同一份 X, y
  -> final CV 重評估
  -> 輸出結果與圖表
```

`quick` preset 預設如下：

```text
proxy_rows = 20000
search_features = 80
lambda = 0.05, 0.10
Random Search trials = 120
GA population = 16
GA generations = 18
SA iterations = 600
inner validation = holdout
final validation = 3-fold CV
```

## 四、資料前處理

正式資料只使用：

```text
data/raw/train_transaction.csv
```

目前沒有合併 `train_identity.csv`。

前處理步驟：

1. 確認 `isFraud` 欄位存在。
2. 如果資料超過 `proxy_rows`，用 stratified sampling 取樣，保留詐欺比例。
3. 移除 `TransactionID`。
4. 數值欄位缺失值補 `-999`。
5. 類別欄位缺失值補 `unknown`，再用 `pandas.factorize()` 編碼。
6. 將 `inf`、`-inf` 轉成 `-999`。
7. 用 `VarianceThreshold` 移除變異數為 0 的欄位。
8. 用 `mutual_info_classif` 排名，取前 `search_features` 個特徵。

注意：MI 預篩只是把搜尋空間縮小，不代表最後一定用 MI 前 K 個特徵。GA/SA 會在這 80 個特徵內再找組合。

## 五、Preprocessing Cache

正式資料前處理很慢，因為 CSV 約 683MB，而且要做類別編碼和 MI。因此程式會 cache 前處理後的資料。

cache 位置：

```text
data/cache/
```

cache 內容：

```text
*.npz   -> X 和 y
*.json  -> feature_names、rows、原始特徵數、參數 metadata
```

cache key 會包含：

```text
train_transaction.csv 的路徑
檔案大小
檔案修改時間
proxy_rows
search_features
random_seed
前處理版本
```

所以換資料、改 seed、改特徵數或改前處理版本時，會自動重建 cache。

如果要強制不用 cache：

```powershell
python run_experiment.py --mode real --data-path .\data\raw\train_transaction.csv --preset quick --no-cache
```

目前實測：

```text
第一次建立 cache：約 57 秒
第二次讀 cache：約 0.02 秒
```

## 六、Fitness 計算

每一個候選解都是一個 binary mask：

```text
1 = 選這個特徵
0 = 不選這個特徵
```

例如搜尋空間有 80 個特徵，mask 長度就是 80。

fitness 公式：

```text
fitness = AUC - lambda * (selected_features / total_features)
```

其中：

```text
AUC = XGBoost 評估出的 ROC-AUC
lambda = 特徵數懲罰係數
selected_features = 這次 mask 選到的特徵數
total_features = 搜尋空間特徵數，quick 是 80
```

lambda 越大，程式越偏好少特徵。

inner loop 為了速度，`quick` preset 使用 holdout validation。最後輸出前會再用 3-fold CV 重新評估每個方法的最佳特徵組合。

## 七、Baseline 方法

目前 baseline 有三種：

```text
AllSearchFeatures
```

使用 MI 預篩後的全部特徵。quick 是 80 個特徵。

```text
RandomSearch
```

隨機產生特徵 mask，跑指定次數後取 fitness 最佳者。

```text
TopK_MI_K
```

取 Mutual Information 排名前 K 個特徵。K 會對齊 GA/SA 最後選到的特徵數，方便比較「同樣特徵數下，單純 MI 排名前 K 是否足夠」。

## 八、GA 流程

GA 在 `optimizers.py` 的 `run_ga()` 中。

流程：

1. 建立初始 population。
2. 每個個體是一個 binary mask。
3. 對每個 mask 計算 fitness。
4. 用 tournament selection 選父母。
5. 用 one-point crossover 產生子代。
6. 用 bit flip mutation 做突變。
7. 保留 elite 個體。
8. 每一代記錄目前 best fitness、AUC 和特徵數。

quick 設定：

```text
population = 16
generations = 18
elitism = 2
tournament_k = 3
crossover_rate = 0.85
mutation_rate = 1 / search_features
```

## 九、SA 流程

SA 在 `optimizers.py` 的 `run_sa()` 中。

流程：

1. 隨機產生初始 binary mask。
2. 每次翻轉 1 到 3 個 bit 產生鄰近解。
3. 如果新解比較好，接受。
4. 如果新解比較差，仍有機率接受。
5. 溫度逐步下降。
6. 記錄目前 best fitness、AUC、特徵數和溫度。

最大化 fitness 時，較差解接受機率是：

```text
exp((new_fitness - old_fitness) / T)
```

quick 設定：

```text
iterations = 600
alpha = 0.985
T_min = 1e-4
log_interval = 50
```

## 十、進度顯示

執行時會看到類似：

```text
RandomSearch 60/120: best_auc=..., features=...
GA generation 10/18: best_auc=..., best_fitness=..., features=...
SA iteration 300/600: best_auc=..., best_fitness=..., features=..., T=...
Final CV 4/12: SA lambda=0.05
```

這代表目前不是卡住，而是在跑對應階段。

## 十一、輸出結果

每次執行會建立：

```text
results/run_YYYYMMDD_HHMMSS/
```

裡面包含：

```text
summary.md
metrics.csv
selected_features.json
convergence.csv
plots/convergence.png
plots/pareto_auc_vs_features.png
plots/lambda_sensitivity.png
```

各檔案用途：

```text
summary.md
```

人類可讀摘要，列出 preset、資料量、特徵數，以及每個 lambda 的最佳方法。

```text
metrics.csv
```

最重要的數字表。包含 inner AUC、inner fitness、final CV AUC、final CV fitness、特徵數和耗時。

```text
selected_features.json
```

記錄每個方法最後選到哪些特徵。

```text
convergence.csv
```

記錄 Random Search、GA、SA 搜尋過程中的 best fitness 變化。

```text
plots/convergence.png
```

GA/SA 收斂曲線。

```text
plots/pareto_auc_vs_features.png
```

比較 AUC 和特徵數。通常用來看是否能用更少特徵達到接近全特徵的效果。

```text
plots/lambda_sensitivity.png
```

比較 lambda 變大時，AUC 和特徵數如何變化。

## 十二、如何判斷結果是否合理

先看 `metrics.csv` 的：

```text
final_cv_auc
final_cv_fitness
selected_count
```

不要只看 inner loop 的 `auc`，因為 inner validation 可能被 GA/SA 搜尋過程過度利用。

合理現象：

1. lambda 變大時，選到的特徵數通常下降。
2. GA/SA 的 convergence curve 應該逐步上升或持平。
3. final CV AUC 可能低於 inner AUC，這是正常的。
4. 如果 GA/SA 用少很多特徵達到接近 AllSearchFeatures 的 final AUC，就可以說特徵篩選有效。
5. 如果 TopK_MI 表現差，代表單變量 MI 排名不足以取代組合式搜尋。

目前最新可參考的正式 quick 結果是：

```text
results/run_20260611_151800/
```

那版重點是：

```text
lambda = 0.05：RandomSearch final AUC 約 0.7346，21 個特徵
lambda = 0.10：GA final fitness 最好，final AUC 約 0.7316，17 個特徵
SA 在 lambda = 0.10 的 final AUC 約 0.7321，22 個特徵
AllSearchFeatures final AUC 約 0.7323，80 個特徵
```

這表示 GA/SA 可以用約 17 到 28 個特徵達到接近 80 個特徵的 AUC，但目前 quick run 還不能宣稱 GA 或 SA 絕對勝出。

## 十三、下一步如果要做得更像正式報告

建議順序：

1. 保留 `quick` 結果當初步實驗。
2. 多跑幾個 seed，觀察平均和標準差。
3. 或改用 `standard` preset 跑更大搜尋預算。
4. 最後整理表格與圖表，寫入期末報告。

可用指令：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_multi_seed_quick.ps1
```

因為 preprocessing cache 已經啟用，同樣資料與相同參數會快很多；但不同 seed 會產生不同 cache，因為 stratified sampling 會不同。

如果已經有多個 `results/run_*`，也可以手動彙整：

```powershell
python aggregate_results.py .\results\run_20260611_153207 .\results\run_另一個結果資料夾
```

彙總輸出會在：

```text
results/aggregate_YYYYMMDD_HHMMSS/
```

裡面最重要的是：

```text
aggregate_summary.md
aggregate_metrics.csv
all_metrics.csv
runs.csv
```
