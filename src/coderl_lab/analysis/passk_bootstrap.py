from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any


def pass_at_k(n: int, c: int, k: int) -> float:
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def task_passk(summary: dict[str, Any], k: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for task_id, row in summary["per_task"].items():
        n = int(row["samples"])
        c = int(row["correct"])
        out[str(task_id)] = pass_at_k(n, c, k)
    return out


def bootstrap_delta(
    a: dict[str, float],
    b: dict[str, float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    ids = sorted(set(a) & set(b))
    if not ids:
        raise ValueError("no overlapping task ids")
    diffs = [b[i] - a[i] for i in ids]
    observed = mean(diffs)

    rng = random.Random(seed)
    samples: list[float] = []
    n = len(diffs)
    for _ in range(iterations):
        draw = [diffs[rng.randrange(n)] for _ in range(n)]
        samples.append(mean(draw))
    samples.sort()

    lo_idx = int(0.025 * iterations)
    hi_idx = min(iterations - 1, int(0.975 * iterations))
    positive = sum(x > 0 for x in samples) / iterations
    return {
        "tasks": n,
        "observed_delta": observed,
        "ci95_low": samples[lo_idx],
        "ci95_high": samples[hi_idx],
        "bootstrap_probability_positive": positive,
        "iterations": iterations,
        "seed": seed,
    }


def compare(
    a_summary: dict[str, Any],
    b_summary: dict[str, Any],
    *,
    ks: list[int],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in ks:
        out[f"pass@{k}"] = bootstrap_delta(
            task_passk(a_summary, k),
            task_passk(b_summary, k),
            iterations=iterations,
            seed=seed,
        )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired bootstrap for Pass@k deltas")
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--k", type=int, nargs="+", required=True)
    p.add_argument("--iterations", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    a = json.loads(args.a.read_text(encoding="utf-8"))
    b = json.loads(args.b.read_text(encoding="utf-8"))
    result = compare(
        a,
        b,
        ks=list(args.k),
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
