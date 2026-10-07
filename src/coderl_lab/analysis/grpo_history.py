from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


METRICS = (
    "reward",
    "reward_std",
    "frac_reward_zero_std",
    "entropy",
    "grad_norm",
    "completions/mean_length",
    "step_time",
    "reward/public_mean",
    "reward/syntax_mean",
    "reward/public_all_pass_mean",
    "reward/dependency_complete_mean",
    "reward/runtime_clean_mean",
)


def _finite(values: list[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out.append(float(value))
    return out


def _mean_for(rows: list[dict[str, Any]], key: str) -> float | None:
    values = _finite([row.get(key) for row in rows])
    return mean(values) if values else None


def summarize_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    steps = [row for row in history if "reward" in row]
    if not steps:
        raise ValueError("log history contains no GRPO reward steps")

    n = len(steps)
    quarter = max(1, n // 4)
    first = steps[:quarter]
    last = steps[-quarter:]

    overall = {key: _mean_for(steps, key) for key in METRICS}
    first_q = {key: _mean_for(first, key) for key in METRICS}
    last_q = {key: _mean_for(last, key) for key in METRICS}

    zero_grad_steps = sum(
        1
        for row in steps
        if isinstance(row.get("grad_norm"), (int, float))
        and float(row["grad_norm"]) == 0.0
    )
    zero_reward_std_steps = sum(
        1
        for row in steps
        if isinstance(row.get("reward_std"), (int, float))
        and float(row["reward_std"]) == 0.0
    )

    return {
        "num_logged_steps": n,
        "overall_mean": overall,
        "first_quarter_mean": first_q,
        "last_quarter_mean": last_q,
        "delta_last_minus_first": {
            key: (
                last_q[key] - first_q[key]
                if last_q[key] is not None and first_q[key] is not None
                else None
            )
            for key in METRICS
        },
        "zero_grad_steps": zero_grad_steps,
        "zero_grad_fraction": zero_grad_steps / n,
        "zero_reward_std_steps": zero_reward_std_steps,
        "zero_reward_std_step_fraction": zero_reward_std_steps / n,
        "reward_min": min(_finite([row.get("reward") for row in steps])),
        "reward_max": max(_finite([row.get("reward") for row in steps])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize CodeRL-Lab GRPO log history"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history = json.loads(args.input.read_text(encoding="utf-8"))
    summary = summarize_history(history)
    text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
