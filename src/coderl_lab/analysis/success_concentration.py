from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pvariance
from typing import Any


def summarize_success_concentration(
    summary: dict[str, Any],
) -> dict[str, Any]:
    rows = list(summary["per_task"].values())
    if not rows:
        raise ValueError("summary has no per-task rows")

    counts = [int(row["correct"]) for row in rows]
    total = sum(counts)
    variance = pvariance(counts)
    shares = [count / total for count in counts] if total else [0.0] * len(counts)
    hhi = sum(share * share for share in shares)

    return {
        "tasks": len(counts),
        "total_correct_candidates": total,
        "mean_correct_per_task": mean(counts),
        "variance_correct_per_task": variance,
        "std_correct_per_task": math.sqrt(variance),
        "zero_correct_tasks": sum(count == 0 for count in counts),
        "all_samples_correct_tasks": sum(
            count == int(row["samples"])
            for count, row in zip(counts, rows, strict=True)
        ),
        "hhi_success_mass": hhi,
        "effective_task_count_from_hhi": 1.0 / hhi if hhi else 0.0,
        "min_correct_per_task": min(counts),
        "max_correct_per_task": max(counts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize how correct-sample mass is concentrated across tasks"
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    result = summarize_success_concentration(summary)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
