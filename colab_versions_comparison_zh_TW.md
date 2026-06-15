# GA/SA Colab 版本完整比較表

## 版本來源

| 版本 | 來源 | 本地解析檔 | 定位 |
|---|---|---|---|
| Colab A | https://colab.research.google.com/drive/1Yi5uBE0-YDbZ_37qGZBSeoyWoBulcRn5 | `external_versions/version_a_1Yi5uBE0.ipynb` | 程式實作較多，後段有多次追加實驗；但預設是 debug，小心不要直接用。 |
| Colab B | https://colab.research.google.com/drive/1uKj7IgmkHR67GW9TCPQpYVPyHSyEe-EY?usp=sharing | `external_versions/version_b_1uKj7Igm.ipynb` | 報告結構最完整，章節、圖表、完整資料 final evaluation、穩定性分析都比較清楚。 |
| 我們版 | `GA_SA_feature_selection_all_in_one_zh_TW.ipynb` | `GA_SA_feature_selection_all_in_one_zh_TW.ipynb` | 已整成單一 ipynb，可直接跑完整 Word strict 流程並產生中文 HTML。 |

## 總結結論

| 項目 | 判斷 |
|---|---|
| 最接近 Word 需求的完整流程 | 我們版。因為預設 `word_strict`，包含 20,000 proxy、Top 150、GA 40/60、SA 5000、inner 5-fold、完整 train_transaction final 5-fold、多 seed、Jaccard、四圖與中文 HTML。 |
| 最適合拿來補報告文字架構 | Colab B。章節完整，包含「資料載入、前處理、GA、SA、Baseline、Final Evaluation、λ 敏感度、視覺化、穩定性」。 |
| 最適合拿來補程式細節 | Colab A 與 B 都可參考。A 有 K=GA/K=SA 的 Top-K MI 實作與 Jaccard；B 的圖表與完整資料 final evaluation 比較完整。 |
| 不建議直接採用的部分 | A 的 `DEBUG_MODE=True` 預設會只跑 GA 12/5、SA 200、Random 100；A 的 final evaluation 只抽 100,000 筆，不是完整 590,540 筆。B 的後段有硬編碼 seed=42 的既有數值，重跑時可能不同步；B 在 lambda/stability 段使用 SA `alpha=0.999`，和 Word 的 `0.98` 不一致。 |
| 建議整合策略 | 用我們版當主執行檔；吸收 B 的章節敘述與圖表命名；保留 A/B 的比較文字作為報告說明；避免採用 A 的 debug/default 與 B 的硬編碼結果。 |

## Word 需求逐項比較

| Word 需求/檢查項 | Colab A | Colab B | 我們 all-in-one | 採用建議 |
|---|---|---|---|---|
| 題目背景與研究問題 | 有零散 markdown，章節不完整 | 完整標題與目錄，較適合報告 | 有中文說明與 Word 對照 | 報告文字可參考 B，我們版保留為主。 |
| 資料集 | 使用 Kaggle IEEE-CIS `train_transaction.csv` | 使用 `ieee-fraud-detection/train_transaction.csv` | 使用 `data/raw/train_transaction.csv` | 三者都符合。 |
| 是否合併 identity 表 | 沒有合併 | 沒有合併 | 沒有合併 | 符合 Word「不合併 identity」。 |
| 原始資料列數 | 讀完整 transaction，但搜尋前抽 proxy | 讀完整 transaction，保留 full 資料 | 讀完整 transaction，proxy 與 final 分開 | 我們版與 B 的資料分工較清楚。 |
| Proxy Dataset | `PROXY_SAMPLE_SIZE = 20000` | `PROXY_SIZE = 20_000` | `PROXY_ROWS = 20_000` | 三者都符合。 |
| Stratified Sampling | 使用 `train_test_split(... stratify=y)` | 使用 `StratifiedShuffleSplit` | 使用 `train_test_split(... stratify=...)` | 三者都符合。 |
| 缺失值處理 | 數值 `-999`、類別 `unknown` | 數值 `-999`、類別 `unknown` | 數值 `-999`、類別 `unknown` | 三者都符合。 |
| 類別編碼 | `OrdinalEncoder`，可套用到 final sample | `LabelEncoder` per column，並存 pkl | `pd.factorize(sort=True)` | B 的 encoder 保存較清楚；我們版較輕量。 |
| Variance Threshold | 有 | 有 | 有 | 三者都符合。 |
| Mutual Information Top 150 | 有，`N_FEATURES_SEARCH = 150` | 有，`TOP_K_FEAT = 150` | 有，`SEARCH_FEATURES = 150` | 三者都符合。 |
| Fitness 公式 | `AUC - lambda * selected/total` | `AUC - lambda * selected/total` | `AUC - lambda * selected/total` | 三者都符合。 |
| Fitness 的 AUC 評估 | `StratifiedKFold n_splits=5` | `StratifiedKFold n_splits=5` | `word_strict` 設 `inner_cv_folds=5` | 三者都可符合；我們版是正式預設。 |
| XGBoost | 有 `XGBClassifier` | 有 `XGBClassifier` | 有 `XGBClassifier` | 三者都符合。 |
| λ 值 | 定義 `[0.0, 0.05, 0.10, 0.20]`，後段有全 λ loop | 主實驗 0.05，後段補 0.0/0.10/0.20 | `lambdas=(0.0,0.05,0.10,0.20)` | 我們版最乾淨；B 可作報告說明。 |
| GA chromosome | binary vector 長度 150 | binary vector 長度 150 | binary mask 長度 150 | 三者都符合。 |
| GA population | 正式分支 40，但 `DEBUG_MODE=True` 預設 12 | 函式與呼叫使用 40 | `word_strict` 使用 40 | A 必須關 debug；我們/B 可直接用。 |
| GA generations | 正式分支 60，但 `DEBUG_MODE=True` 預設 5 | 使用 60 | `word_strict` 使用 60 | A 必須關 debug；我們/B 可直接用。 |
| GA selection | Tournament k=3 | Tournament k=3 | Tournament k=3 | 三者都符合。 |
| GA crossover | One-point, rate 0.85 | One-point, rate 0.85 | One-point, rate 0.85 | 三者都符合。 |
| GA mutation | `1/n_features` | 預設 `1/n_features` | 固定 `1/150` | 三者符合 Word。 |
| GA elitism | 有 elite 保留 | 有 elite 保留 | `ga_elites=2` | 三者都符合。 |
| SA state | binary vector | binary vector | binary mask | 三者都符合。 |
| SA 初始解 | 隨機 | 隨機 | 隨機 | 三者都符合。 |
| SA neighbor move | 隨機翻轉 1-3 bit | `n_flips=1` | 隨機翻轉 1-3 bit | A/我們較貼近 Word。B 可接受但較簡化。 |
| SA 初始溫度 | 有校準 | 有 `sa_calibrate_T0` | 有 calibration moves | 三者都符合。 |
| SA calibration moves | A 程式有校準，但不是明確 100 次 | B 預設 calibration function 內 `n_moves=100` | `sa_calibration_moves=100` | 我們/B 較明確。 |
| SA alpha | A 使用 0.98 | B 主段 0.98，但 lambda/stability 段用 0.999 | `word_strict` 使用 0.98 | 我們最一致；B 需修正 0.999。 |
| SA iterations | 正式分支 5000，但 debug 預設 200 | 使用 5000 | `word_strict` 使用 5000 | A 必須關 debug；我們/B 可直接用。 |
| Random Search | 有；正式分支 2400 evals | 有；2400 evals，等於 GA 40x60 | 有；`word_strict` 用 5000 evals | Word 說同樣 evaluations，B/A 對 GA 公平；我們對 SA 也更保守。 |
| All Features baseline | 有，全部 150 | 有，全部 150 | 有，`AllSearchFeatures` | 三者都符合。 |
| Top-K by MI | 有 K=GA 與 K=SA | 有 K=GA | 有 K=GA 與 K=SA count | A/我們更完整；B 已符合最低需求。 |
| 單一 λ 主實驗 | 有 λ=0.05 | 有 λ=0.05 | 多 λ 都跑 | 我們版不只單一 λ。 |
| λ 敏感度 | 後段有全 λ，但程式有重複追加 | 有補跑 0.0/0.10/0.20 加上 0.05 | 每個 seed 跑全 λ | 我們最一致；B 結構可參考。 |
| Final evaluation dataset | 抽 `FINAL_SAMPLE_SIZE=100000`，排除 proxy 後抽樣 | 完整 `X_full_eval`，約 590K | 完整 `train_transaction.csv`，約 590K | A 不符合完整訓練集；B/我們符合。 |
| Final evaluation CV | 5-fold CV | 5-fold CV | 5-fold CV | B/我們完整；A 是 100K final sample 的 5-fold。 |
| Final evaluation 是否不含懲罰 | 有 final_auc | 有 `最終AUC(Full)` 不含懲罰 | 有 `final_cv_auc` 與 `final_cv_fitness` | B/我們較清楚。 |
| Proxy vs Final 差異 | 有 proxy vs final 圖與表 | 有 proxy vs full 誤差說明 | metrics 保留 inner/final 欄位 | B/A 的報告敘述可補到報告；我們資料欄位完整。 |
| Multi-seed 穩定性 | 沒有完整 multi-seed | 有 seed 42/43/44，但 seed 42 數值硬編碼 | 有 `SEEDS=[7,42,2026]` 自動跑完整流程 | 我們最乾淨；B 可參考呈現格式。 |
| Stability std | 缺 | 有 AUC/std、特徵數/std | aggregate 自動輸出 mean/std | 我們/B 符合；A 缺。 |
| GA/SA Jaccard overlap | 有 | 圖二輸出 Jaccard | 有 per-run 與 aggregate Jaccard | 三者都可支援；我們輸出最完整。 |
| 收斂曲線圖 | 有 | 有 `fig1_convergence.png` | 有 `01_convergence.png` | 三者符合。 |
| 特徵重要性/MI 選取圖 | A 的 MI/feature 圖較不完整或分散 | 有 `fig2_feature_importance.png` | 有 `02_feature_importance_selection.png` | B/我們符合；可用 B 的圖表風格。 |
| Pareto AUC vs features | 有 | 有 `fig3_pareto_front.png` | 有 `03_pareto_auc_vs_features.png` | 三者符合。 |
| λ sensitivity 圖 | 後段有，但檔案與流程重複 | 有 `fig4_lambda_sensitivity.png` | 有 `04_lambda_sensitivity.png` | B/我們較清楚。 |
| 中文圖表字型 | 未特別完整處理 | 設 `Microsoft JhengHei` | matplotlib 圖主要英文標籤，HTML 中文 | 若報告圖要全中文，可借 B 的字型設定。 |
| 結果輸出 | `results/metrics`、selected features、figures | `preprocessed/` 內 CSV/NPY/PNG | `results/run_*` 與 `results/aggregate_*` | 我們輸出結構最適合重跑與彙整。 |
| 快取/中斷保護 | 有 fitness cache；部分存 CSV | 有 fitness cache、joblib、每段存檔 | 有 preprocessing/full-eval cache、fitness cache、run artifacts | 我們/B 較穩。 |
| HTML 彙整 | 無 | 無 | 有 `GA_SA_report_summary_zh_TW.html` | 我們獨有。 |
| 單一 notebook 是否可跑全流程 | 是，但預設 debug 會跑少 | 是，但有硬編碼/需按段落跑 | 是，預設 `RUN_MULTI_SEED=True` 跑完整 strict | 我們最符合目前需求。 |
| 編碼處理 | Colab 環境通常沒問題 | 有中文字型設定 | `.ipynb` raw ASCII JSON，避免中文亂碼 | 我們最適合 Windows/VS Code。 |
| 主要風險 | debug 預設、final 只 100K、沒有 multi-seed、自動化程度低 | 硬編碼結果、SA alpha 0.999 不一致、部分段落需手動順序 | runtime 很長，需要確實 Run All 產生 strict 新結果 | 用我們版跑；從 B 補報告敘述。 |

## 建議整合版內容安排

| 報告/程式區塊 | 建議採用來源 | 原因 |
|---|---|---|
| 主執行 notebook | 我們版 | 已是單一 ipynb，預設正式 strict，全流程自動化。 |
| 章節標題與報告敘述 | Colab B | B 的中文章節最接近期末報告骨架。 |
| 資料前處理說明 | Colab B + 我們版 | B 說明清楚；我們版實作有 cache 與 all-in-one。 |
| GA/SA 參數表 | 我們版 | 完整對齊 Word strict，避免 B 的 alpha 0.999 與 A 的 debug。 |
| Baseline 設計 | 我們版 + Colab A | 我們/A 都含 K=GA 與 K=SA，較完整。 |
| Final evaluation | 我們版 | B 符合完整資料，但有硬編碼結果；我們是從每次 run 自動產生。 |
| 四張圖 | 我們版 + Colab B 視覺風格 | 我們自動輸出完整圖；B 的中文圖表命名與字型設定可參考。 |
| 穩定性分析 | 我們版 | 自動 multi-seed，不需硬編碼 seed=42 既有結果。 |
| 中文 HTML/彙整 | 我們版 | A/B 沒有 HTML 彙整。 |

## 最終採用建議

1. 用 `GA_SA_feature_selection_all_in_one_zh_TW.ipynb` 作為唯一可執行版本。
2. 報告文字可吸收 Colab B 的章節順序：環境、資料、前處理、MI、Proxy、Fitness、GA、SA、Baseline、Final Evaluation、λ 敏感度、視覺化、穩定性。
3. 不要直接用 Colab A 的預設設定，因為 `DEBUG_MODE=True` 會跑少。
4. 不要直接沿用 Colab A 的 final evaluation 結論，因為它是 100,000 筆 final sample，不是完整 590,540 筆。
5. 如果參考 Colab B 的 lambda/stability 段，需把 SA `alpha=0.999` 改回 Word 的 `0.98`，並移除硬編碼 seed=42 結果，改成讀實際輸出。
6. 正式交付結果應以我們版重新 Run All 後的 `results/aggregate_*`、`figures_for_report/`、`GA_SA_report_summary_zh_TW.html` 為準。
