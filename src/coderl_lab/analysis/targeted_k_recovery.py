from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _solved(summary: dict[str, Any], task_id: str) -> bool:
    return int(summary["per_task"][task_id]["correct"]) > 0


def _correct(summary: dict[str, Any], task_id: str) -> int:
    return int(summary["per_task"][task_id]["correct"])


def analyze_targeted_recovery(
    *,
    original_targets: list[str],
    base: dict[str, Any],
    sft: dict[str, Any],
    grpo: dict[str, Any],
) -> dict[str, Any]:
    target_set = set(original_targets)
    available = set(base["per_task"]) & set(sft["per_task"]) & set(grpo["per_task"])
    if target_set != available:
        missing = sorted(target_set - available)
        extra = sorted(available - target_set)
        raise ValueError(f"target mismatch: missing={missing}, extra={extra}")

    rows = []
    for task_id in sorted(original_targets):
        rows.append(
            {
                "task_id": task_id,
                "base_correct_64": _correct(base, task_id),
                "sft_correct_64": _correct(sft, task_id),
                "grpo_correct_64": _correct(grpo, task_id),
            }
        )

    sft_recovered_64 = [r["task_id"] for r in rows if r["sft_correct_64"] > 0]
    sft_still_missing_64 = [r["task_id"] for r in rows if r["sft_correct_64"] == 0]
    grpo_recovered_64 = [r["task_id"] for r in rows if r["grpo_correct_64"] > 0]
    grpo_recovers_sft_missing_64 = [
        r["task_id"]
        for r in rows
        if r["sft_correct_64"] == 0 and r["grpo_correct_64"] > 0
    ]

    return {
        "target_count": len(rows),
        "sft_recovered_at_64": len(sft_recovered_64),
        "sft_recovery_fraction_at_64": (
            len(sft_recovered_64) / len(rows) if rows else 1.0
        ),
        "sft_still_missing_at_64": len(sft_still_missing_64),
        "sft_still_missing_task_ids": sft_still_missing_64,
        "grpo_solved_at_64": len(grpo_recovered_64),
        "grpo_recovers_sft_missing_at_64": len(grpo_recovers_sft_missing_64),
        "grpo_recovers_sft_missing_task_ids": grpo_recovers_sft_missing_64,
        "per_task": rows,
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze targeted n=64 recovery")
    p.add_argument("--n16-support", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--sft", type=Path, required=True)
    p.add_argument("--grpo", type=Path, required=True)
    p.add_argument("--output", type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    support = _load(args.n16_support)
    result = analyze_targeted_recovery(
        original_targets=list(support["task_ids"]["base_lost_by_sft"]),
        base=_load(args.base),
        sft=_load(args.sft),
        grpo=_load(args.grpo),
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
