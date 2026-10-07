from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any


CONDITIONS = ("baseline", "low_stabilize", "low_destabilize", "high_stabilize")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_effects(arms: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    task_ids = sorted(
        set.intersection(
            *[set(arm["per_task"]) for arm in arms]
        )
    )
    out: dict[str, dict[str, float]] = {}
    for task_id in task_ids:
        rows = [arm["per_task"][task_id] for arm in arms]
        eligible_rows = [row for row in rows if bool(row["eligible"])]
        if not eligible_rows:
            continue
        values: dict[str, float] = {}
        for condition in CONDITIONS:
            vals = [
                float(bool(row[condition]["exact_match_reference"]))
                for row in eligible_rows
            ]
            values[condition] = mean(vals)
        out[task_id] = values
    return out


def _cluster_bootstrap(
    task_effects: dict[str, dict[str, float]],
    *,
    left: str,
    right: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    ids = sorted(task_effects)
    diffs = [
        task_effects[t][right] - task_effects[t][left]
        for t in ids
    ]
    observed = mean(diffs) if diffs else 0.0
    rng = random.Random(seed)
    draws = []
    n = len(diffs)
    for _ in range(iterations):
        draw = mean(diffs[rng.randrange(n)] for _ in range(n))
        draws.append(draw)
    draws.sort()
    lo = draws[int(0.025 * iterations)]
    hi = draws[min(iterations - 1, int(0.975 * iterations))]
    return {
        "tasks": n,
        "observed_delta": observed,
        "ci95_low": lo,
        "ci95_high": hi,
        "probability_positive": sum(x > 0 for x in draws) / iterations,
        "iterations": iterations,
        "seed": seed,
    }


def summarize(
    arms: list[dict[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    task_effects = _task_effects(arms)

    pair_rows = []
    for arm in arms:
        for task_id, row in arm["per_task"].items():
            if not bool(row["eligible"]):
                continue
            pair_rows.append(row)

    exact = {
        condition: mean(
            float(bool(row[condition]["exact_match_reference"]))
            for row in pair_rows
        )
        for condition in CONDITIONS
    }

    baseline_divergent = [
        row for row in pair_rows
        if not bool(row["baseline"]["exact_match_reference"])
    ]
    baseline_matching = [
        row for row in pair_rows
        if bool(row["baseline"]["exact_match_reference"])
    ]

    rescue = {
        "baseline_divergent_pairs": len(baseline_divergent),
        "low_stabilize_rescued": sum(
            bool(row["low_stabilize"]["exact_match_reference"])
            for row in baseline_divergent
        ),
        "high_stabilize_rescued": sum(
            bool(row["high_stabilize"]["exact_match_reference"])
            for row in baseline_divergent
        ),
    }
    if baseline_divergent:
        rescue["low_rescue_fraction"] = (
            rescue["low_stabilize_rescued"]
            / len(baseline_divergent)
        )
        rescue["high_rescue_fraction"] = (
            rescue["high_stabilize_rescued"]
            / len(baseline_divergent)
        )

    induced = {
        "baseline_matching_pairs": len(baseline_matching),
        "low_destabilize_induced": sum(
            not bool(row["low_destabilize"]["exact_match_reference"])
            for row in baseline_matching
        ),
    }
    if baseline_matching:
        induced["induced_fraction"] = (
            induced["low_destabilize_induced"]
            / len(baseline_matching)
        )

    applied = {
        condition: mean(
            float(bool(row[condition].get("intervention_applied", False)))
            for row in pair_rows
        )
        for condition in ("low_stabilize", "low_destabilize", "high_stabilize")
    }

    return {
        "arms": len(arms),
        "eligible_tasks": len(task_effects),
        "eligible_task_arm_pairs": len(pair_rows),
        "exact_match_fraction": exact,
        "intervention_applied_fraction": applied,
        "rescue": rescue,
        "induced_divergence": induced,
        "task_cluster_bootstrap": {
            "low_stabilize_minus_baseline": _cluster_bootstrap(
                task_effects,
                left="baseline",
                right="low_stabilize",
                iterations=iterations,
                seed=seed,
            ),
            "high_stabilize_minus_baseline": _cluster_bootstrap(
                task_effects,
                left="baseline",
                right="high_stabilize",
                iterations=iterations,
                seed=seed,
            ),
            "low_stabilize_minus_high_control": _cluster_bootstrap(
                task_effects,
                left="high_stabilize",
                right="low_stabilize",
                iterations=iterations,
                seed=seed,
            ),
            "low_destabilize_minus_baseline": _cluster_bootstrap(
                task_effects,
                left="baseline",
                right="low_destabilize",
                iterations=iterations,
                seed=seed,
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = summarize(
        [_load(path) for path in args.arm],
        iterations=args.iterations,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
