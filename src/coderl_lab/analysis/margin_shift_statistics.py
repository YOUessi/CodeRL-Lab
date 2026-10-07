from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any

from coderl_lab.analysis.susceptibility_statistics import bootstrap_spearman


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("values must be non-empty")
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    weight = pos - lo
    return float(
        ordered[lo] * (1.0 - weight)
        + ordered[hi] * weight
    )


def paired_mean_bootstrap(
    base: list[float],
    sft: list[float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if len(base) != len(sft) or not base:
        raise ValueError("paired inputs must have equal non-zero length")
    diffs = [b - a for a, b in zip(base, sft, strict=True)]
    observed = mean(diffs)
    rng = random.Random(seed)
    n = len(diffs)
    draws = [
        mean(diffs[rng.randrange(n)] for _ in range(n))
        for _ in range(iterations)
    ]
    draws.sort()
    return {
        "observed_mean_delta_sft_minus_base": observed,
        "ci95_low": _percentile(draws, 0.025),
        "ci95_high": _percentile(draws, 0.975),
        "bootstrap_probability_positive": (
            sum(v > 0 for v in draws) / len(draws)
        ),
        "iterations": iterations,
        "seed": seed,
        "tasks_increased": sum(d > 0 for d in diffs),
        "tasks_decreased": sum(d < 0 for d in diffs),
        "tasks_equal": sum(d == 0 for d in diffs),
    }


def _feature(row: dict[str, Any], name: str) -> float:
    p = row["first128"]["probability_margin"]
    mapping = {
        "low_margin_fraction_0_01": p["fraction_le_0_01"],
        "low_margin_fraction_0_05": p["fraction_le_0_05"],
        "low_margin_fraction_0_10": p["fraction_le_0_10"],
        "tie_fraction": p["fraction_le_1e_6"],
        "min_margin": p["min"],
        "p10_margin": p["p10"],
        "median_margin": p["median"],
    }
    return float(mapping[name])


def analyze_margin_shift(
    *,
    base_profile: dict[str, Any],
    sft_profile: dict[str, Any],
    susceptibility: dict[str, Any],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    task_ids = sorted(
        set(base_profile["per_task"])
        & set(sft_profile["per_task"])
        & set(susceptibility["per_task"])
    )
    if len(task_ids) != 90:
        raise ValueError(f"expected 90 paired tasks, got {len(task_ids)}")

    features = (
        "low_margin_fraction_0_01",
        "low_margin_fraction_0_05",
        "low_margin_fraction_0_10",
        "tie_fraction",
        "min_margin",
        "p10_margin",
        "median_margin",
    )

    shifts: dict[str, Any] = {}
    feature_deltas: dict[str, list[float]] = {}
    for feature in features:
        base_values = [
            _feature(base_profile["per_task"][task_id], feature)
            for task_id in task_ids
        ]
        sft_values = [
            _feature(sft_profile["per_task"][task_id], feature)
            for task_id in task_ids
        ]
        feature_deltas[feature] = [
            sft - base
            for base, sft in zip(base_values, sft_values, strict=True)
        ]
        shifts[feature] = paired_mean_bootstrap(
            base_values,
            sft_values,
            iterations=iterations,
            seed=seed,
        )
        shifts[feature]["base_mean"] = mean(base_values)
        shifts[feature]["sft_mean"] = mean(sft_values)

    base_lengths = [
        int(base_profile["per_task"][task_id]["completion_token_count"])
        for task_id in task_ids
    ]
    sft_lengths = [
        int(sft_profile["per_task"][task_id]["completion_token_count"])
        for task_id in task_ids
    ]
    length_shift = paired_mean_bootstrap(
        [float(x) for x in base_lengths],
        [float(x) for x in sft_lengths],
        iterations=iterations,
        seed=seed,
    )
    length_shift["base_mean"] = mean(base_lengths)
    length_shift["sft_mean"] = mean(sft_lengths)

    outcome = [
        float(
            susceptibility["per_task"][task_id]["greedy_divergence_rate"]
        )
        for task_id in task_ids
    ]
    primary_delta = feature_deltas["low_margin_fraction_0_05"]
    delta_vs_susceptibility = bootstrap_spearman(
        primary_delta,
        outcome,
        iterations=iterations,
        seed=seed,
    )

    return {
        "tasks": len(task_ids),
        "primary_shift": shifts["low_margin_fraction_0_05"],
        "primary_shift_positive": bool(
            shifts["low_margin_fraction_0_05"]["ci95_low"] > 0
        ),
        "feature_shifts": shifts,
        "completion_length_shift": length_shift,
        "delta_low_margin_vs_susceptibility_spearman": (
            delta_vs_susceptibility
        ),
        "per_task": {
            task_id: {
                "base_low_margin_fraction_0_05": _feature(
                    base_profile["per_task"][task_id],
                    "low_margin_fraction_0_05",
                ),
                "sft_low_margin_fraction_0_05": _feature(
                    sft_profile["per_task"][task_id],
                    "low_margin_fraction_0_05",
                ),
                "delta_low_margin_fraction_0_05": (
                    feature_deltas["low_margin_fraction_0_05"][index]
                ),
                "greedy_divergence_rate": outcome[index],
                "base_completion_length": base_lengths[index],
                "sft_completion_length": sft_lengths[index],
            }
            for index, task_id in enumerate(task_ids)
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-profile", type=Path, required=True)
    parser.add_argument("--sft-profile", type=Path, required=True)
    parser.add_argument("--susceptibility", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = json.loads(args.base_profile.read_text(encoding="utf-8"))
    sft = json.loads(args.sft_profile.read_text(encoding="utf-8"))
    susceptibility = json.loads(
        args.susceptibility.read_text(encoding="utf-8")
    )
    result = analyze_margin_shift(
        base_profile=base,
        sft_profile=sft,
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
                "primary_shift": result["primary_shift"],
                "primary_shift_positive": result[
                    "primary_shift_positive"
                ],
                "length_shift": result["completion_length_shift"],
                "delta_vs_susceptibility": result[
                    "delta_low_margin_vs_susceptibility_spearman"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
