from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def prepare_extension(
    *,
    support_analysis: dict[str, Any],
    tasks: list[dict[str, Any]],
    base_predictions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    target_ids = list(
        support_analysis["task_ids"]["base_lost_by_sft"]
    )
    target_set = set(target_ids)

    selected = [row for row in tasks if str(row["task_id"]) in target_set]
    selected_ids = [str(row["task_id"]) for row in selected]

    missing = target_set - set(selected_ids)
    if missing:
        raise ValueError(
            "target task ids missing from validation tasks: "
            + ", ".join(sorted(missing))
        )

    seed_map: dict[str, int] = {}
    for row in base_predictions:
        task_id = str(row["task_id"])
        if task_id not in target_set:
            continue
        if int(row["sample_id"]) != 0:
            continue
        generation = dict(row.get("generation", {}))
        if "seed" not in generation:
            raise ValueError(f"prediction for {task_id} has no generation seed")
        seed_map[task_id] = int(generation["seed"])

    missing_seeds = target_set - set(seed_map)
    if missing_seeds:
        raise ValueError(
            "target task ids missing sample-0 seeds: "
            + ", ".join(sorted(missing_seeds))
        )

    return selected, seed_map


def verify_prefix(
    *,
    original_predictions: list[dict[str, Any]],
    extended_predictions: list[dict[str, Any]],
    target_ids: set[str],
    prefix_samples: int,
) -> dict[str, Any]:
    original = {
        (str(row["task_id"]), int(row["sample_id"])): str(row["completion"])
        for row in original_predictions
        if str(row["task_id"]) in target_ids
        and int(row["sample_id"]) < prefix_samples
    }
    extended = {
        (str(row["task_id"]), int(row["sample_id"])): str(row["completion"])
        for row in extended_predictions
        if str(row["task_id"]) in target_ids
        and int(row["sample_id"]) < prefix_samples
    }

    missing = sorted(set(original) - set(extended))
    mismatched = sorted(
        key
        for key, value in original.items()
        if key in extended and extended[key] != value
    )
    return {
        "prefix_samples": prefix_samples,
        "expected_rows": len(original),
        "matched_rows": len(original) - len(missing) - len(mismatched),
        "missing_rows": [
            {"task_id": task_id, "sample_id": sample_id}
            for task_id, sample_id in missing
        ],
        "mismatched_rows": [
            {"task_id": task_id, "sample_id": sample_id}
            for task_id, sample_id in mismatched
        ],
        "identical": not missing and not mismatched,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare targeted n=64 continuation from EXP-006B n=16"
    )
    p.add_argument("--support-analysis", type=Path, required=True)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--base-predictions", type=Path, required=True)
    p.add_argument("--output-tasks", type=Path, required=True)
    p.add_argument("--output-seeds", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    support = json.loads(args.support_analysis.read_text(encoding="utf-8"))
    tasks = load_jsonl(args.tasks)
    predictions = load_jsonl(args.base_predictions)

    selected, seed_map = prepare_extension(
        support_analysis=support,
        tasks=tasks,
        base_predictions=predictions,
    )
    write_jsonl(args.output_tasks, selected)
    args.output_seeds.parent.mkdir(parents=True, exist_ok=True)
    args.output_seeds.write_text(
        json.dumps(seed_map, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "target_tasks": len(selected),
                "task_ids": [row["task_id"] for row in selected],
                "seed_map_entries": len(seed_map),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
