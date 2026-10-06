from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _correct(summary: dict[str, Any], task_id: str) -> int:
    return int(summary["per_task"][task_id]["correct"])


def track_k4_losses(
    *,
    base_k4: dict[str, Any],
    sft_k4: dict[str, Any],
    base_n16: dict[str, Any],
    sft_n16: dict[str, Any],
    grpo_n16: dict[str, Any],
) -> dict[str, Any]:
    ids = sorted(set(base_k4["per_task"]) & set(sft_k4["per_task"]))
    lost_at_k4 = [
        task_id
        for task_id in ids
        if _correct(base_k4, task_id) > 0
        and _correct(sft_k4, task_id) == 0
    ]

    recovered_by_sft_n16 = [
        task_id for task_id in lost_at_k4
        if _correct(sft_n16, task_id) > 0
    ]
    still_missing_sft_n16 = [
        task_id for task_id in lost_at_k4
        if _correct(sft_n16, task_id) == 0
    ]
    recovered_by_grpo_n16 = [
        task_id for task_id in lost_at_k4
        if _correct(grpo_n16, task_id) > 0
    ]
    grpo_recovers_sft_missing_n16 = [
        task_id for task_id in still_missing_sft_n16
        if _correct(grpo_n16, task_id) > 0
    ]
    base_resolved_n16 = [
        task_id for task_id in lost_at_k4
        if _correct(base_n16, task_id) > 0
    ]

    return {
        "k4_lost_task_count": len(lost_at_k4),
        "k4_lost_task_ids": lost_at_k4,
        "base_resolved_at_n16": len(base_resolved_n16),
        "sft_recovered_at_n16": len(recovered_by_sft_n16),
        "sft_recovery_fraction": (
            len(recovered_by_sft_n16) / len(lost_at_k4)
            if lost_at_k4 else 1.0
        ),
        "sft_still_missing_at_n16": len(still_missing_sft_n16),
        "sft_still_missing_task_ids": still_missing_sft_n16,
        "grpo_solved_at_n16": len(recovered_by_grpo_n16),
        "grpo_recovers_sft_missing_at_n16": len(
            grpo_recovers_sft_missing_n16
        ),
        "grpo_recovers_sft_missing_task_ids": grpo_recovers_sft_missing_n16,
        "note": (
            "The n=16 run is a unified resampling experiment, not a strict "
            "prefix extension of the earlier k=4 generation. These counts "
            "track the same task identities under a larger fresh sample budget."
        ),
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Track EXP-006A k=4 lost tasks under EXP-006B n=16"
    )
    p.add_argument("--base-k4", type=Path, required=True)
    p.add_argument("--sft-k4", type=Path, required=True)
    p.add_argument("--base-n16", type=Path, required=True)
    p.add_argument("--sft-n16", type=Path, required=True)
    p.add_argument("--grpo-n16", type=Path, required=True)
    p.add_argument("--output", type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = track_k4_losses(
        base_k4=_load(args.base_k4),
        sft_k4=_load(args.sft_k4),
        base_n16=_load(args.base_n16),
        sft_n16=_load(args.sft_n16),
        grpo_n16=_load(args.grpo_n16),
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
