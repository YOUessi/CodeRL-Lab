from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_predictions(
    rows: list[dict[str, Any]],
    *,
    expected_tasks: int,
    expected_samples: int,
) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()

    for row in rows:
        task_id = str(row["task_id"])
        sample_id = int(row["sample_id"])
        key = (task_id, sample_id)
        if key in seen:
            raise ValueError(f"duplicate prediction key: {key}")
        seen.add(key)
        grouped[task_id].append(sample_id)

    if len(grouped) != expected_tasks:
        raise ValueError(
            f"expected {expected_tasks} tasks, got {len(grouped)}"
        )

    expected_ids = list(range(expected_samples))
    bad: dict[str, list[int]] = {}
    for task_id, sample_ids in grouped.items():
        ids = sorted(sample_ids)
        if ids != expected_ids:
            bad[task_id] = ids

    if bad:
        preview = list(bad.items())[:5]
        raise ValueError(
            f"tasks do not have exact sample ids 0..{expected_samples - 1}: "
            f"{preview}"
        )

    expected_rows = expected_tasks * expected_samples
    if len(rows) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} rows, got {len(rows)}"
        )

    return {
        "tasks": len(grouped),
        "samples_per_task": expected_samples,
        "rows": len(rows),
        "task_ids": sorted(grouped),
    }


def validate_three(
    *,
    base: list[dict[str, Any]],
    sft: list[dict[str, Any]],
    grpo: list[dict[str, Any]],
    expected_tasks: int,
    expected_samples: int,
) -> dict[str, Any]:
    summaries = {
        "base": validate_predictions(
            base,
            expected_tasks=expected_tasks,
            expected_samples=expected_samples,
        ),
        "sft": validate_predictions(
            sft,
            expected_tasks=expected_tasks,
            expected_samples=expected_samples,
        ),
        "grpo": validate_predictions(
            grpo,
            expected_tasks=expected_tasks,
            expected_samples=expected_samples,
        ),
    }
    task_sets = {
        name: set(summary["task_ids"])
        for name, summary in summaries.items()
    }
    if not (
        task_sets["base"] == task_sets["sft"] == task_sets["grpo"]
    ):
        raise ValueError("base/sft/grpo task id sets differ")

    for summary in summaries.values():
        summary.pop("task_ids", None)

    return {
        "valid": True,
        "expected_tasks": expected_tasks,
        "expected_samples": expected_samples,
        "models": summaries,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate EXP-006B n=16 prediction artifacts"
    )
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--sft", type=Path, required=True)
    p.add_argument("--grpo", type=Path, required=True)
    p.add_argument("--expected-tasks", type=int, default=90)
    p.add_argument("--expected-samples", type=int, default=16)
    p.add_argument("--output", type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_three(
        base=load_jsonl(args.base),
        sft=load_jsonl(args.sft),
        grpo=load_jsonl(args.grpo),
        expected_tasks=args.expected_tasks,
        expected_samples=args.expected_samples,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
