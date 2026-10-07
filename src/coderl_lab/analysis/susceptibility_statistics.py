from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


def _rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for offset in range(start, end):
            ranks[indexed[offset][0]] = average_rank
        start = end
    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("x and y must have equal length >= 2")
    mx = mean(x)
    my = mean(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    denom = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    if denom == 0:
        raise ValueError("correlation undefined for constant input")
    return sum(a * b for a, b in zip(dx, dy, strict=True)) / denom


def spearman(x: list[float], y: list[float]) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("values must be non-empty")
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    weight = pos - lo
    return float(
        sorted_values[lo] * (1.0 - weight)
        + sorted_values[hi] * weight
    )


def bootstrap_spearman(
    x: list[float],
    y: list[float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    observed = spearman(x, y)
    rng = random.Random(seed)
    n = len(x)
    draws: list[float] = []
    for _ in range(iterations):
        idx = [rng.randrange(n) for _ in range(n)]
        bx = [x[i] for i in idx]
        by = [y[i] for i in idx]
        try:
            draws.append(spearman(bx, by))
        except ValueError:
            continue
    draws.sort()
    if not draws:
        raise ValueError("all bootstrap correlation draws were degenerate")
    return {
        "observed_rho": observed,
        "ci95_low": _percentile(draws, 0.025),
        "ci95_high": _percentile(draws, 0.975),
        "bootstrap_probability_positive": (
            sum(v > 0 for v in draws) / len(draws)
        ),
        "valid_bootstrap_draws": len(draws),
        "iterations_requested": iterations,
        "seed": seed,
    }


def _standardize(values: np.ndarray) -> np.ndarray:
    std = float(values.std())
    if std == 0:
        raise ValueError("cannot standardize constant values")
    return (values - values.mean()) / std


def adjusted_ols(
    predictor: list[float],
    lengths: list[int],
    outcome: list[float],
) -> dict[str, float]:
    x = _standardize(np.asarray(predictor, dtype=np.float64))
    length = _standardize(
        np.log1p(np.asarray(lengths, dtype=np.float64))
    )
    y = np.asarray(outcome, dtype=np.float64)
    design = np.column_stack([np.ones(len(y)), x, length])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return {
        "intercept": float(beta[0]),
        "low_margin_standardized_coefficient": float(beta[1]),
        "log_length_standardized_coefficient": float(beta[2]),
    }


def bootstrap_adjusted_ols(
    predictor: list[float],
    lengths: list[int],
    outcome: list[float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    observed = adjusted_ols(predictor, lengths, outcome)
    rng = random.Random(seed)
    n = len(outcome)
    draws: list[float] = []
    for _ in range(iterations):
        idx = [rng.randrange(n) for _ in range(n)]
        try:
            row = adjusted_ols(
                [predictor[i] for i in idx],
                [lengths[i] for i in idx],
                [outcome[i] for i in idx],
            )
        except ValueError:
            continue
        draws.append(float(row["low_margin_standardized_coefficient"]))
    draws.sort()
    if not draws:
        raise ValueError("all OLS bootstrap draws were degenerate")
    return {
        **observed,
        "low_margin_coefficient_ci95_low": _percentile(draws, 0.025),
        "low_margin_coefficient_ci95_high": _percentile(draws, 0.975),
        "bootstrap_probability_positive": (
            sum(v > 0 for v in draws) / len(draws)
        ),
        "valid_bootstrap_draws": len(draws),
        "iterations_requested": iterations,
        "seed": seed,
    }


def quartile_contrast(
    predictor: list[float],
    outcome: list[float],
) -> dict[str, Any]:
    order = sorted(range(len(predictor)), key=lambda i: predictor[i])
    size = max(1, len(order) // 4)
    bottom = order[:size]
    top = order[-size:]
    bottom_mean = mean(outcome[i] for i in bottom)
    top_mean = mean(outcome[i] for i in top)
    return {
        "quartile_size": size,
        "bottom_predictor_mean": mean(predictor[i] for i in bottom),
        "top_predictor_mean": mean(predictor[i] for i in top),
        "bottom_outcome_mean": bottom_mean,
        "top_outcome_mean": top_mean,
        "top_minus_bottom": top_mean - bottom_mean,
    }


def _feature(
    row: dict[str, Any],
    name: str,
) -> float:
    probability = row["first128"]["probability_margin"]
    mapping = {
        "low_margin_fraction_0_01": probability["fraction_le_0_01"],
        "low_margin_fraction_0_05": probability["fraction_le_0_05"],
        "low_margin_fraction_0_10": probability["fraction_le_0_10"],
        "tie_fraction": probability["fraction_le_1e_6"],
        "p10_margin": probability["p10"],
        "median_margin": probability["median"],
        "min_margin": probability["min"],
    }
    return float(mapping[name])


def analyze(
    *,
    margin_profile: dict[str, Any],
    susceptibility: dict[str, Any],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    task_ids = sorted(
        set(margin_profile["per_task"]) & set(susceptibility["per_task"])
    )
    if len(task_ids) != 90:
        raise ValueError(f"expected 90 tasks, got {len(task_ids)}")

    primary = [
        _feature(
            margin_profile["per_task"][task_id],
            "low_margin_fraction_0_05",
        )
        for task_id in task_ids
    ]
    lengths = [
        int(margin_profile["per_task"][task_id]["completion_token_count"])
        for task_id in task_ids
    ]
    outcome = [
        float(
            susceptibility["per_task"][task_id]["greedy_divergence_rate"]
        )
        for task_id in task_ids
    ]

    primary_corr = bootstrap_spearman(
        primary,
        outcome,
        iterations=iterations,
        seed=seed,
    )
    length_control = bootstrap_adjusted_ols(
        primary,
        lengths,
        outcome,
        iterations=iterations,
        seed=seed,
    )

    secondary: dict[str, Any] = {}
    for feature in (
        "low_margin_fraction_0_01",
        "low_margin_fraction_0_10",
        "tie_fraction",
        "p10_margin",
        "median_margin",
        "min_margin",
    ):
        values = [
            _feature(margin_profile["per_task"][task_id], feature)
            for task_id in task_ids
        ]
        secondary[feature] = bootstrap_spearman(
            values,
            outcome,
            iterations=iterations,
            seed=seed,
        )

    by_scale: dict[str, Any] = {}
    for scale in ("0.25", "0.5", "1.0", "2.0"):
        scale_outcome = [
            float(
                susceptibility["per_task"][task_id]["by_scale"][scale][
                    "divergence_rate"
                ]
            )
            for task_id in task_ids
        ]
        by_scale[scale] = bootstrap_spearman(
            primary,
            scale_outcome,
            iterations=iterations,
            seed=seed,
        )

    result = {
        "tasks": len(task_ids),
        "primary_predictor": "first128 probability margin fraction <= 0.05",
        "primary_outcome": "12-arm greedy divergence rate",
        "primary_spearman": primary_corr,
        "primary_supported": bool(primary_corr["ci95_low"] > 0),
        "length_adjusted_ols": length_control,
        "quartile_contrast": quartile_contrast(primary, outcome),
        "secondary_spearman": secondary,
        "by_scale_primary_spearman": by_scale,
        "descriptives": {
            "mean_primary_predictor": mean(primary),
            "mean_greedy_divergence_rate": mean(outcome),
            "mean_completion_length": mean(lengths),
        },
        "per_task": {
            task_id: {
                "low_margin_fraction_0_05": primary[index],
                "completion_token_count": lengths[index],
                "greedy_divergence_rate": outcome[index],
            }
            for index, task_id in enumerate(task_ids)
        },
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin-profile", type=Path, required=True)
    parser.add_argument("--susceptibility", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    margin = json.loads(args.margin_profile.read_text(encoding="utf-8"))
    susceptibility = json.loads(
        args.susceptibility.read_text(encoding="utf-8")
    )
    result = analyze(
        margin_profile=margin,
        susceptibility=susceptibility,
        iterations=args.iterations,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "primary_spearman": result["primary_spearman"],
                "primary_supported": result["primary_supported"],
                "length_adjusted_ols": result["length_adjusted_ols"],
                "quartile_contrast": result["quartile_contrast"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
