from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass(frozen=True)
class FitnessResult:
    fitness: float
    auc: float
    selected_count: int
    selected_ratio: float
    seconds: float
    cached: bool


class FitnessEvaluator:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        penalty_lambda: float,
        cv_folds: int,
        test_size: float,
        n_estimators: int,
        max_depth: int,
        learning_rate: float,
        device: str,
        random_seed: int,
        n_jobs: int = 8,
        use_cache: bool = True,
    ) -> None:
        self.X = X
        self.y = y
        self.penalty_lambda = penalty_lambda
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.device = device
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.use_cache = use_cache
        self.cache: dict[str, FitnessResult] = {}
        self.model_fits = 0

    @property
    def total_features(self) -> int:
        return self.X.shape[1]

    def evaluate(self, mask: np.ndarray) -> FitnessResult:
        mask = normalize_mask(mask)
        key = pack_mask(mask)
        if self.use_cache and key in self.cache:
            cached = self.cache[key]
            return FitnessResult(
                fitness=cached.fitness,
                auc=cached.auc,
                selected_count=cached.selected_count,
                selected_ratio=cached.selected_ratio,
                seconds=0.0,
                cached=True,
            )

        selected_count = int(mask.sum())
        selected_ratio = selected_count / self.total_features
        if selected_count == 0:
            result = FitnessResult(
                fitness=0.5,
                auc=0.5,
                selected_count=0,
                selected_ratio=0.0,
                seconds=0.0,
                cached=False,
            )
            if self.use_cache:
                self.cache[key] = result
            return result

        t0 = perf_counter()
        X_sub = self.X[:, mask]
        auc = self._score_auc(X_sub)
        seconds = perf_counter() - t0
        fitness = auc - self.penalty_lambda * selected_ratio
        result = FitnessResult(
            fitness=float(fitness),
            auc=float(auc),
            selected_count=selected_count,
            selected_ratio=float(selected_ratio),
            seconds=seconds,
            cached=False,
        )
        if self.use_cache:
            self.cache[key] = result
        return result

    def _score_auc(self, X_sub: np.ndarray) -> float:
        if self.cv_folds <= 1:
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_sub,
                self.y,
                test_size=self.test_size,
                stratify=self.y,
                random_state=self.random_seed,
            )
            return self._fit_predict_auc(X_train, y_train, X_valid, y_valid)

        splitter = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_seed,
        )
        aucs: list[float] = []
        for train_idx, valid_idx in splitter.split(X_sub, self.y):
            aucs.append(
                self._fit_predict_auc(
                    X_sub[train_idx],
                    self.y[train_idx],
                    X_sub[valid_idx],
                    self.y[valid_idx],
                )
            )
        return float(np.mean(aucs))

    def _fit_predict_auc(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
    ) -> float:
        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "auc",
            "random_state": self.random_seed,
            "n_jobs": self.n_jobs,
            "tree_method": "hist",
        }
        if self.device == "cuda":
            params["device"] = "cuda"

        pos = max(1, int(np.sum(y_train == 1)))
        neg = max(1, int(np.sum(y_train == 0)))
        params["scale_pos_weight"] = neg / pos

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        self.model_fits += 1
        pred = model.predict_proba(X_valid)[:, 1]
        return float(roc_auc_score(y_valid, pred))


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    clean = np.asarray(mask, dtype=bool).copy()
    if clean.ndim != 1:
        raise ValueError("Feature mask must be one-dimensional.")
    return clean


def pack_mask(mask: np.ndarray) -> str:
    return "".join("1" if value else "0" for value in mask.astype(bool))
