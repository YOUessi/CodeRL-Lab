from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

STRATA = tuple(
    (platform, difficulty)
    for platform in ("atcoder", "leetcode")
    for difficulty in ("easy", "medium", "hard")
)


def select_smoke_tasks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fixed, outcome-blind 2x6 smoke covering both LCB execution modes."""
    if len({str(row["question_id"]) for row in rows}) != len(rows):
        raise ValueError("duplicate LiveCodeBench question ID")

    picked: list[dict[str, Any]] = []
    for platform, difficulty in STRATA:
        candidates = sorted(
            (
                row for row in rows
                if row["platform"] == platform and row["difficulty"] == difficulty
            ),
            key=lambda row: str(row["question_id"]),
        )
        if len(candidates) < 2:
            raise ValueError(
                f"need two {platform}/{difficulty} smoke tasks, found {len(candidates)}"
            )
        picked.extend(candidates[:2])

    ids = [str(row["question_id"]) for row in picked]
    if len(picked) != 12 or len(set(ids)) != 12:
        raise AssertionError("12-task smoke selection is invalid")

    types = Counter(
        str(test["testtype"])
        for row in picked
        for test in row["public_test_cases"]
    )
    if not {"stdin", "functional"}.issubset(types):
        raise AssertionError("both stdin and functional tasks are required")
    return picked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from coderl_lab.analysis.livecodebench_official_eval import load_public_tasks

    rows = list(load_public_tasks(args.public_tasks).values())
    if len(rows) != 175:
        raise ValueError(f"expected 175 external tasks, got {len(rows)}")
    selected = select_smoke_tasks(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "smoke_tasks": len(selected),
        "selection": "2 lexicographically earliest IDs per platform/difficulty",
        "question_ids": [str(row["question_id"]) for row in selected],
        "source_private_tests_read": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
