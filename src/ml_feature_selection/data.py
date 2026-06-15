from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.model_selection import train_test_split


@dataclass
class PreparedData:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    selected_from_original: list[str]
    source: str
    rows: int
    original_features: int


CACHE_VERSION = "preprocess-v2-mi-desc"


def load_synthetic(proxy_rows: int, random_seed: int) -> tuple[pd.DataFrame, pd.Series]:
    n_features = 150
    X, y = make_classification(
        n_samples=proxy_rows,
        n_features=n_features,
        n_informative=35,
        n_redundant=20,
        n_repeated=0,
        n_clusters_per_class=2,
        weights=[0.965, 0.035],
        flip_y=0.01,
        class_sep=1.1,
        random_state=random_seed,
    )
    names = [f"f{i:03d}" for i in range(n_features)]
    return pd.DataFrame(X, columns=names), pd.Series(y, name="isFraud")


def load_ieee_transaction(path: Path, proxy_rows: int, random_seed: int) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到資料檔：{path}\n"
            "請先把 Kaggle IEEE-CIS 的 train_transaction.csv 放到 data/raw/。"
        )

    df = pd.read_csv(path, low_memory=False)
    if "isFraud" not in df.columns:
        raise ValueError("train_transaction.csv 中找不到 isFraud 欄位。")

    if proxy_rows and len(df) > proxy_rows:
        _, df = train_test_split(
            df,
            test_size=proxy_rows,
            stratify=df["isFraud"],
            random_state=random_seed,
        )

    y = df.pop("isFraud").astype("int8")
    for col in ("TransactionID",):
        if col in df.columns:
            df = df.drop(columns=[col])
    return df.reset_index(drop=True), y.reset_index(drop=True)


def prepare_ieee_transaction_with_cache(
    path: Path,
    proxy_rows: int,
    search_features: int,
    random_seed: int,
    cache_dir: Path,
) -> tuple[PreparedData, bool, Path]:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到資料檔：{path}\n"
            "請先把 Kaggle IEEE-CIS 的 train_transaction.csv 放到 data/raw/。"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(path, proxy_rows, search_features, random_seed)
    npz_path = cache_dir / f"{key}.npz"
    meta_path = cache_dir / f"{key}.json"

    if npz_path.exists() and meta_path.exists():
        return _load_prepared_cache(npz_path, meta_path), True, npz_path

    X_raw, y = load_ieee_transaction(path, proxy_rows, random_seed)
    data = preprocess_and_select(
        X_raw=X_raw,
        y=y,
        search_features=search_features,
        random_seed=random_seed,
        source=str(path),
    )
    _save_prepared_cache(data, npz_path, meta_path, path, proxy_rows, search_features, random_seed)
    return data, False, npz_path


def preprocess_and_select(
    X_raw: pd.DataFrame,
    y: pd.Series,
    search_features: int,
    random_seed: int,
    source: str,
) -> PreparedData:
    original_features = X_raw.shape[1]
    X = X_raw.copy()

    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    if numeric_cols:
        X[numeric_cols] = X[numeric_cols].fillna(-999)
    for col in categorical_cols:
        values = X[col].astype("string").fillna("unknown")
        X[col] = pd.factorize(values, sort=True)[0].astype("int32")

    X = X.replace([np.inf, -np.inf], -999)
    X_np = X.to_numpy(dtype=np.float32, copy=True)
    y_np = y.to_numpy(dtype=np.int8, copy=True)
    names = X.columns.tolist()

    variance = VarianceThreshold(threshold=0.0)
    X_var = variance.fit_transform(X_np)
    kept_names = [name for name, keep in zip(names, variance.get_support()) if keep]
    if X_var.shape[1] == 0:
        raise ValueError("VarianceThreshold 後沒有可用特徵。")

    k = min(search_features, X_var.shape[1])
    mi = mutual_info_classif(X_var, y_np, discrete_features=False, random_state=random_seed)
    top_idx = np.argsort(mi)[::-1][:k]

    X_selected = X_var[:, top_idx].astype(np.float32, copy=False)
    selected_names = [kept_names[i] for i in top_idx]

    return PreparedData(
        X=X_selected,
        y=y_np,
        feature_names=selected_names,
        selected_from_original=selected_names,
        source=source,
        rows=X_selected.shape[0],
        original_features=original_features,
    )


def _cache_key(path: Path, proxy_rows: int, search_features: int, random_seed: int) -> str:
    stat = path.stat()
    payload = {
        "version": CACHE_VERSION,
        "path": str(path.resolve()).lower(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "proxy_rows": proxy_rows,
        "search_features": search_features,
        "random_seed": random_seed,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _save_prepared_cache(
    data: PreparedData,
    npz_path: Path,
    meta_path: Path,
    raw_path: Path,
    proxy_rows: int,
    search_features: int,
    random_seed: int,
) -> None:
    np.savez(npz_path, X=data.X, y=data.y)
    metadata = {
        "cache_version": CACHE_VERSION,
        "raw_path": str(raw_path),
        "proxy_rows": proxy_rows,
        "search_features": search_features,
        "random_seed": random_seed,
        "feature_names": data.feature_names,
        "selected_from_original": data.selected_from_original,
        "source": data.source,
        "rows": data.rows,
        "original_features": data.original_features,
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_prepared_cache(npz_path: Path, meta_path: Path) -> PreparedData:
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as arrays:
        X = arrays["X"]
        y = arrays["y"]
    return PreparedData(
        X=X,
        y=y,
        feature_names=list(metadata["feature_names"]),
        selected_from_original=list(metadata["selected_from_original"]),
        source=str(metadata["source"]),
        rows=int(metadata["rows"]),
        original_features=int(metadata["original_features"]),
    )
