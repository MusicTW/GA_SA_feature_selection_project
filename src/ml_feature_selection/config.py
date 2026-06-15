from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal


PresetName = Literal["smoke", "quick", "standard"]


@dataclass(frozen=True)
class ExperimentConfig:
    preset: PresetName
    proxy_rows: int
    search_features: int
    lambdas: tuple[float, ...]
    random_trials: int
    ga_population: int
    ga_generations: int
    ga_elites: int
    ga_tournament_k: int
    ga_crossover_rate: float
    ga_mutation_rate: float | None
    sa_iterations: int
    sa_log_interval: int
    sa_alpha: float
    sa_t_min: float
    sa_calibration_moves: int
    inner_cv_folds: int
    final_cv_folds: int
    test_size: float
    xgb_estimators_inner: int
    xgb_estimators_final: int
    xgb_max_depth: int
    xgb_learning_rate: float
    random_seed: int


PRESETS: dict[PresetName, ExperimentConfig] = {
    "smoke": ExperimentConfig(
        preset="smoke",
        proxy_rows=3_000,
        search_features=30,
        lambdas=(0.05,),
        random_trials=20,
        ga_population=8,
        ga_generations=5,
        ga_elites=2,
        ga_tournament_k=3,
        ga_crossover_rate=0.85,
        ga_mutation_rate=None,
        sa_iterations=80,
        sa_log_interval=10,
        sa_alpha=0.98,
        sa_t_min=1e-4,
        sa_calibration_moves=10,
        inner_cv_folds=1,
        final_cv_folds=3,
        test_size=0.2,
        xgb_estimators_inner=30,
        xgb_estimators_final=60,
        xgb_max_depth=3,
        xgb_learning_rate=0.08,
        random_seed=42,
    ),
    "quick": ExperimentConfig(
        preset="quick",
        proxy_rows=20_000,
        search_features=80,
        lambdas=(0.05, 0.10),
        random_trials=120,
        ga_population=16,
        ga_generations=18,
        ga_elites=2,
        ga_tournament_k=3,
        ga_crossover_rate=0.85,
        ga_mutation_rate=None,
        sa_iterations=600,
        sa_log_interval=50,
        sa_alpha=0.985,
        sa_t_min=1e-4,
        sa_calibration_moves=25,
        inner_cv_folds=1,
        final_cv_folds=3,
        test_size=0.2,
        xgb_estimators_inner=60,
        xgb_estimators_final=120,
        xgb_max_depth=4,
        xgb_learning_rate=0.08,
        random_seed=42,
    ),
    "standard": ExperimentConfig(
        preset="standard",
        proxy_rows=20_000,
        search_features=120,
        lambdas=(0.0, 0.05, 0.10, 0.20),
        random_trials=300,
        ga_population=24,
        ga_generations=30,
        ga_elites=2,
        ga_tournament_k=3,
        ga_crossover_rate=0.85,
        ga_mutation_rate=None,
        sa_iterations=1_500,
        sa_log_interval=100,
        sa_alpha=0.988,
        sa_t_min=1e-4,
        sa_calibration_moves=40,
        inner_cv_folds=1,
        final_cv_folds=5,
        test_size=0.2,
        xgb_estimators_inner=80,
        xgb_estimators_final=160,
        xgb_max_depth=4,
        xgb_learning_rate=0.08,
        random_seed=42,
    ),
}


def get_config(name: PresetName, **overrides: object) -> ExperimentConfig:
    config = PRESETS[name]
    clean = {key: value for key, value in overrides.items() if value is not None}
    return replace(config, **clean) if clean else config
