from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(
    *,
    baseline: dict[str, Any],
    arms: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metrics = ("pass@1", "pass@4", "pass@8", "pass@16")
    rows: dict[str, Any] = {}

    for name, arm in arms.items():
        row = {
            "pass_at_k": arm["pass_at_k"],
            "solved_tasks": int(arm["solved_tasks"]),
            "hidden_mean_pass_rate": float(arm["hidden_mean_pass_rate"]),
            "delta_vs_baseline": {},
        }
        for metric in metrics:
            row["delta_vs_baseline"][metric] = (
                float(arm["pass_at_k"][metric])
                - float(baseline["pass_at_k"][metric])
            )
        row["delta_vs_baseline"]["solved_tasks"] = (
            int(arm["solved_tasks"]) - int(baseline["solved_tasks"])
        )
        rows[name] = row

    aggregate: dict[str, Any] = {}
    for metric in metrics:
        values = [
            rows[name]["delta_vs_baseline"][metric]
            for name in rows
        ]
        aggregate[metric] = {
            "mean_delta": statistics.mean(values),
            "stdev_delta": (
                statistics.stdev(values) if len(values) > 1 else 0.0
            ),
            "min_delta": min(values),
            "max_delta": max(values),
            "positive_seeds": sum(value > 0 for value in values),
            "num_seeds": len(values),
        }

    solved_values = [
        rows[name]["delta_vs_baseline"]["solved_tasks"]
        for name in rows
    ]
    aggregate["solved_tasks"] = {
        "mean_delta": statistics.mean(solved_values),
        "stdev_delta": (
            statistics.stdev(solved_values)
            if len(solved_values) > 1
            else 0.0
        ),
        "min_delta": min(solved_values),
        "max_delta": max(solved_values),
        "positive_seeds": sum(value > 0 for value in solved_values),
        "num_seeds": len(solved_values),
    }

    return {
        "baseline": {
            "pass_at_k": baseline["pass_at_k"],
            "solved_tasks": baseline["solved_tasks"],
            "hidden_mean_pass_rate": baseline["hidden_mean_pass_rate"],
        },
        "arms": rows,
        "aggregate_delta": aggregate,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        help="Repeated NAME=PATH",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arms: dict[str, dict[str, Any]] = {}
    for item in args.arm:
        name, path = item.split("=", 1)
        arms[name] = load(Path(path))

    result = summarize(
        baseline=load(args.baseline),
        arms=arms,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
