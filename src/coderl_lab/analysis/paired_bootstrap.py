from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def task_success_rates(summary: dict[str, Any]) -> dict[str, float]:
    return {
        task_id: float(row["correct"]) / int(row["samples"])
        for task_id, row in summary["per_task"].items()
    }


def task_solved(summary: dict[str, Any]) -> dict[str, float]:
    return {
        task_id: 1.0 if int(row["correct"]) > 0 else 0.0
        for task_id, row in summary["per_task"].items()
    }


def paired_bootstrap_delta(
    a: dict[str, float],
    b: dict[str, float],
    *,
    iterations: int = 20000,
    seed: int = 42,
) -> dict[str, float]:
    ids = sorted(set(a) & set(b))
    if not ids:
        raise ValueError("no overlapping task ids")

    diffs = [b[i] - a[i] for i in ids]
    observed = sum(diffs) / len(diffs)

    rng = random.Random(seed)
    boots: list[float] = []
    n = len(diffs)
    for _ in range(iterations):
        boots.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    boots.sort()

    lo = boots[int(0.025 * iterations)]
    hi = boots[min(iterations - 1, int(0.975 * iterations))]
    prob_positive = sum(x > 0 for x in boots) / iterations

    return {
        "tasks": n,
        "observed_delta": observed,
        "ci95_low": lo,
        "ci95_high": hi,
        "bootstrap_probability_positive": prob_positive,
        "iterations": iterations,
        "seed": seed,
    }


def compare_summaries(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    iterations: int = 20000,
    seed: int = 42,
) -> dict[str, Any]:
    return {
        "pass_at_1_task_rate_delta": paired_bootstrap_delta(
            task_success_rates(a),
            task_success_rates(b),
            iterations=iterations,
            seed=seed,
        ),
        "pass_at_4_solved_task_delta": paired_bootstrap_delta(
            task_solved(a),
            task_solved(b),
            iterations=iterations,
            seed=seed,
        ),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired bootstrap between two evaluations")
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    a = json.loads(args.a.read_text(encoding="utf-8"))
    b = json.loads(args.b.read_text(encoding="utf-8"))
    result = compare_summaries(
        a,
        b,
        iterations=args.iterations,
        seed=args.seed,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
