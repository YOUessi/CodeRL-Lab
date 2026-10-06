from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .paired_bootstrap import paired_bootstrap_delta, task_success_rates


def task_solved_at_n(summary: dict[str, Any]) -> dict[str, float]:
    return {
        task_id: 1.0 if int(row["correct"]) > 0 else 0.0
        for task_id, row in summary["per_task"].items()
    }


def compare_support(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    iterations: int = 20000,
    seed: int = 42,
) -> dict[str, Any]:
    sample_counts = {
        int(row["samples"])
        for row in a["per_task"].values()
    } | {
        int(row["samples"])
        for row in b["per_task"].values()
    }
    if len(sample_counts) != 1:
        raise ValueError("all tasks in both summaries must share one sample count")
    n = next(iter(sample_counts))

    return {
        "samples_per_task": n,
        "empirical_success_rate_delta": paired_bootstrap_delta(
            task_success_rates(a),
            task_success_rates(b),
            iterations=iterations,
            seed=seed,
        ),
        "solved_at_n_delta": paired_bootstrap_delta(
            task_solved_at_n(a),
            task_solved_at_n(b),
            iterations=iterations,
            seed=seed,
        ),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paired bootstrap for empirical success and solved@n support"
    )
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
    result = compare_support(
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
