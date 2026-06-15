from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from time import perf_counter
from typing import Callable

import numpy as np

from .evaluation import FitnessEvaluator, FitnessResult, normalize_mask


@dataclass
class SearchResult:
    method: str
    penalty_lambda: float
    mask: np.ndarray
    result: FitnessResult
    seconds: float
    history: list[dict[str, float | int | str]]


def ensure_non_empty(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mask = normalize_mask(mask)
    if not mask.any():
        mask[int(rng.integers(0, len(mask)))] = True
    return mask


def random_mask(n_features: int, rng: np.random.Generator, p: float = 0.5) -> np.ndarray:
    return ensure_non_empty(rng.random(n_features) < p, rng)


def run_random_search(
    evaluator: FitnessEvaluator,
    trials: int,
    rng: np.random.Generator,
    penalty_lambda: float,
    progress_callback: Callable[[str], None] | None = None,
) -> SearchResult:
    t0 = perf_counter()
    best_mask = random_mask(evaluator.total_features, rng)
    best_result = evaluator.evaluate(best_mask)
    history: list[dict[str, float | int | str]] = []

    for step in range(1, trials + 1):
        p = float(rng.uniform(0.15, 0.65))
        mask = random_mask(evaluator.total_features, rng, p=p)
        result = evaluator.evaluate(mask)
        if result.fitness > best_result.fitness:
            best_mask = mask
            best_result = result
        history.append(
            _history_row(
                method="RandomSearch",
                penalty_lambda=penalty_lambda,
                step=step,
                best=best_result,
                avg_fitness=result.fitness,
            )
        )
        if progress_callback and _should_report(step, trials):
            progress_callback(
                f"RandomSearch {step}/{trials}: "
                f"best_auc={best_result.auc:.4f}, features={best_result.selected_count}"
            )

    return SearchResult(
        method="RandomSearch",
        penalty_lambda=penalty_lambda,
        mask=best_mask,
        result=best_result,
        seconds=perf_counter() - t0,
        history=history,
    )


def run_ga(
    evaluator: FitnessEvaluator,
    population_size: int,
    generations: int,
    elites: int,
    tournament_k: int,
    crossover_rate: float,
    mutation_rate: float,
    rng: np.random.Generator,
    penalty_lambda: float,
    progress_callback: Callable[[str], None] | None = None,
) -> SearchResult:
    t0 = perf_counter()
    n_features = evaluator.total_features
    population = np.array([random_mask(n_features, rng) for _ in range(population_size)])
    scored = [evaluator.evaluate(ind) for ind in population]
    history: list[dict[str, float | int | str]] = []

    for generation in range(1, generations + 1):
        order = np.argsort([res.fitness for res in scored])[::-1]
        population = population[order]
        scored = [scored[i] for i in order]
        next_population = [population[i].copy() for i in range(min(elites, population_size))]

        while len(next_population) < population_size:
            parent_a = _tournament(population, scored, tournament_k, rng)
            parent_b = _tournament(population, scored, tournament_k, rng)
            child_a, child_b = _crossover(parent_a, parent_b, crossover_rate, rng)
            child_a = _mutate(child_a, mutation_rate, rng)
            child_b = _mutate(child_b, mutation_rate, rng)
            next_population.append(ensure_non_empty(child_a, rng))
            if len(next_population) < population_size:
                next_population.append(ensure_non_empty(child_b, rng))

        population = np.array(next_population)
        scored = [evaluator.evaluate(ind) for ind in population]
        best = max(scored, key=lambda res: res.fitness)
        avg = float(np.mean([res.fitness for res in scored]))
        history.append(
            _history_row(
                method="GA",
                penalty_lambda=penalty_lambda,
                step=generation,
                best=best,
                avg_fitness=avg,
            )
        )
        if progress_callback and _should_report(generation, generations):
            progress_callback(
                f"GA generation {generation}/{generations}: "
                f"best_auc={best.auc:.4f}, best_fitness={best.fitness:.4f}, "
                f"features={best.selected_count}"
            )

    best_idx = int(np.argmax([res.fitness for res in scored]))
    return SearchResult(
        method="GA",
        penalty_lambda=penalty_lambda,
        mask=population[best_idx].copy(),
        result=scored[best_idx],
        seconds=perf_counter() - t0,
        history=history,
    )


def run_sa(
    evaluator: FitnessEvaluator,
    iterations: int,
    alpha: float,
    t_min: float,
    calibration_moves: int,
    log_interval: int,
    rng: np.random.Generator,
    penalty_lambda: float,
    progress_callback: Callable[[str], None] | None = None,
) -> SearchResult:
    t0 = perf_counter()
    current = random_mask(evaluator.total_features, rng)
    current_result = evaluator.evaluate(current)
    best = current.copy()
    best_result = current_result
    temperature = _calibrate_temperature(evaluator, current, current_result, calibration_moves, rng)
    history: list[dict[str, float | int | str]] = []

    for iteration in range(1, iterations + 1):
        candidate = _neighbor(current, rng)
        candidate_result = evaluator.evaluate(candidate)
        delta = candidate_result.fitness - current_result.fitness
        if delta >= 0 or rng.random() < exp(delta / max(temperature, 1e-12)):
            current = candidate
            current_result = candidate_result
        if current_result.fitness > best_result.fitness:
            best = current.copy()
            best_result = current_result

        temperature = max(t_min, temperature * alpha)
        if iteration == 1 or iteration % log_interval == 0 or iteration == iterations:
            history.append(
                _history_row(
                    method="SA",
                    penalty_lambda=penalty_lambda,
                    step=iteration,
                    best=best_result,
                    avg_fitness=current_result.fitness,
                    temperature=temperature,
                )
            )
            if progress_callback:
                progress_callback(
                    f"SA iteration {iteration}/{iterations}: "
                    f"best_auc={best_result.auc:.4f}, best_fitness={best_result.fitness:.4f}, "
                    f"features={best_result.selected_count}, T={temperature:.5f}"
                )

    return SearchResult(
        method="SA",
        penalty_lambda=penalty_lambda,
        mask=best,
        result=best_result,
        seconds=perf_counter() - t0,
        history=history,
    )


def _tournament(
    population: np.ndarray,
    scored: list[FitnessResult],
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    idx = rng.choice(len(population), size=min(k, len(population)), replace=False)
    winner = max(idx, key=lambda i: scored[int(i)].fitness)
    return population[int(winner)].copy()


def _crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    crossover_rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if len(parent_a) < 2 or rng.random() >= crossover_rate:
        return parent_a.copy(), parent_b.copy()
    point = int(rng.integers(1, len(parent_a)))
    child_a = np.concatenate([parent_a[:point], parent_b[point:]])
    child_b = np.concatenate([parent_b[:point], parent_a[point:]])
    return child_a, child_b


def _mutate(mask: np.ndarray, mutation_rate: float, rng: np.random.Generator) -> np.ndarray:
    child = mask.copy()
    flips = rng.random(len(child)) < mutation_rate
    child[flips] = ~child[flips]
    return child


def _neighbor(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    candidate = mask.copy()
    flips = int(rng.integers(1, 4))
    idx = rng.choice(len(candidate), size=min(flips, len(candidate)), replace=False)
    candidate[idx] = ~candidate[idx]
    return ensure_non_empty(candidate, rng)


def _calibrate_temperature(
    evaluator: FitnessEvaluator,
    current: np.ndarray,
    current_result: FitnessResult,
    calibration_moves: int,
    rng: np.random.Generator,
) -> float:
    bad_deltas: list[float] = []
    for _ in range(calibration_moves):
        candidate = _neighbor(current, rng)
        result = evaluator.evaluate(candidate)
        delta = result.fitness - current_result.fitness
        if delta < 0:
            bad_deltas.append(abs(delta))
    if not bad_deltas:
        return 0.02
    return max(float(np.mean(bad_deltas)) / -log(0.8), 1e-4)


def _history_row(
    method: str,
    penalty_lambda: float,
    step: int,
    best: FitnessResult,
    avg_fitness: float,
    temperature: float | None = None,
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "method": method,
        "lambda": penalty_lambda,
        "step": step,
        "best_fitness": best.fitness,
        "best_auc": best.auc,
        "best_selected_count": best.selected_count,
        "avg_or_current_fitness": avg_fitness,
    }
    if temperature is not None:
        row["temperature"] = temperature
    return row


def _should_report(step: int, total: int) -> bool:
    interval = max(1, total // 10)
    return step == 1 or step % interval == 0 or step == total
