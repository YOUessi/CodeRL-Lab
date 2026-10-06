from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bootstrap_delta(
    a: list[float],
    b: list[float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if len(a) != len(b) or not a:
        raise ValueError("paired bootstrap requires equal non-empty vectors")

    observed = sum(y - x for x, y in zip(a, b, strict=True)) / len(a)
    rng = random.Random(seed)
    n = len(a)
    samples = []
    for _ in range(iterations):
        idx = [rng.randrange(n) for _ in range(n)]
        delta = sum(b[i] - a[i] for i in idx) / n
        samples.append(delta)
    samples.sort()

    lo = samples[int(0.025 * iterations)]
    hi = samples[min(iterations - 1, int(0.975 * iterations))]
    positive = sum(x > 0 for x in samples) / iterations

    return {
        "tasks": n,
        "observed_delta": observed,
        "ci95_low": lo,
        "ci95_high": hi,
        "bootstrap_probability_positive": positive,
        "iterations": iterations,
        "seed": seed,
    }


def compare_pair(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    ids = sorted(set(a["per_task"]) & set(b["per_task"]))
    if set(a["per_task"]) != set(b["per_task"]):
        raise ValueError("MBPP+ summaries must contain the same task ids")

    plus_a = [float(a["per_task"][i]["plus_success_rate"]) for i in ids]
    plus_b = [float(b["per_task"][i]["plus_success_rate"]) for i in ids]
    solved_a = [1.0 if a["per_task"][i]["plus_correct"] > 0 else 0.0 for i in ids]
    solved_b = [1.0 if b["per_task"][i]["plus_correct"] > 0 else 0.0 for i in ids]

    return {
        "plus_empirical_success_rate": _bootstrap_delta(
            plus_a, plus_b, iterations=iterations, seed=seed
        ),
        "plus_solved_at_n": _bootstrap_delta(
            solved_a, solved_b, iterations=iterations, seed=seed
        ),
    }


def model_summary(summary: dict[str, Any]) -> dict[str, Any]:
    base_total = sum(v["base_correct"] for v in summary["per_task"].values())
    plus_total = sum(v["plus_correct"] for v in summary["per_task"].values())
    return {
        "base_pass_at_k": summary["base_pass_at_k"],
        "plus_pass_at_k": summary["plus_pass_at_k"],
        "base_solved_tasks": summary["base_solved_tasks"],
        "plus_solved_tasks": summary["plus_solved_tasks"],
        "mean_base_success_rate": summary["mean_base_success_rate"],
        "mean_plus_success_rate": summary["mean_plus_success_rate"],
        "mean_robustness_gap": summary["mean_robustness_gap"],
        "candidate_base_correct": base_total,
        "candidate_plus_correct": plus_total,
        "candidate_robust_retention": (
            plus_total / base_total if base_total else None
        ),
    }


def compare(
    *,
    base: dict[str, Any],
    sft: dict[str, Any],
    grpo: dict[str, Any],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "models": {
            "base": model_summary(base),
            "sft": model_summary(sft),
            "grpo": model_summary(grpo),
        },
        "paired_bootstrap": {
            "sft_minus_base": compare_pair(
                base, sft, iterations=iterations, seed=seed
            ),
            "grpo_minus_sft": compare_pair(
                sft, grpo, iterations=iterations, seed=seed
            ),
            "grpo_minus_base": compare_pair(
                base, grpo, iterations=iterations, seed=seed
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare Base/SFT/GRPO on MBPP+")
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--sft", type=Path, required=True)
    p.add_argument("--grpo", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = compare(
        base=_load(args.base),
        sft=_load(args.sft),
        grpo=_load(args.grpo),
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
