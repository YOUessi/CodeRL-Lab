from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def solved_set(summary: dict[str, Any]) -> set[str]:
    return {
        task_id
        for task_id, row in summary["per_task"].items()
        if int(row["correct"]) > 0
    }


def all_correct_set(summary: dict[str, Any]) -> set[str]:
    return {
        task_id
        for task_id, row in summary["per_task"].items()
        if int(row["correct"]) == int(row["samples"])
    }


def mean_empirical_success(summary: dict[str, Any]) -> float:
    rows = list(summary["per_task"].values())
    return sum(
        int(row["correct"]) / int(row["samples"])
        for row in rows
    ) / len(rows)


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def compare_large_k(
    *,
    base: dict[str, Any],
    sft: dict[str, Any],
    grpo: dict[str, Any],
) -> dict[str, Any]:
    task_ids = set(base["per_task"])
    if task_ids != set(sft["per_task"]) or task_ids != set(grpo["per_task"]):
        raise ValueError("base/sft/grpo task sets must match")

    b = solved_set(base)
    s = solved_set(sft)
    g = solved_set(grpo)

    base_lost_by_sft = b - s
    base_new_in_sft = s - b
    recovered_by_grpo = base_lost_by_sft & g
    sft_lost_by_grpo = s - g
    grpo_new_vs_sft = g - s

    def model_summary(summary: dict[str, Any]) -> dict[str, Any]:
        solved = solved_set(summary)
        all_correct = all_correct_set(summary)
        return {
            "pass_at_k": dict(summary["pass_at_k"]),
            "solved_at_n": len(solved),
            "all_n_correct_tasks": len(all_correct),
            "mean_empirical_success_rate": mean_empirical_success(summary),
        }

    return {
        "num_tasks": len(task_ids),
        "samples_per_task": next(iter(base["per_task"].values()))["samples"],
        "models": {
            "base": model_summary(base),
            "sft": model_summary(sft),
            "grpo": model_summary(grpo),
        },
        "support_sets": {
            "base_solved": len(b),
            "sft_solved": len(s),
            "grpo_solved": len(g),
            "base_retained_by_sft": len(b & s),
            "base_retention_fraction_sft": len(b & s) / len(b) if b else 1.0,
            "base_lost_by_sft": len(base_lost_by_sft),
            "sft_new_vs_base": len(base_new_in_sft),
            "base_lost_then_recovered_by_grpo": len(recovered_by_grpo),
            "recovery_fraction_grpo": (
                len(recovered_by_grpo) / len(base_lost_by_sft)
                if base_lost_by_sft
                else 1.0
            ),
            "sft_lost_by_grpo": len(sft_lost_by_grpo),
            "grpo_new_vs_sft": len(grpo_new_vs_sft),
            "jaccard_base_sft": jaccard(b, s),
            "jaccard_sft_grpo": jaccard(s, g),
            "jaccard_base_grpo": jaccard(b, g),
        },
        "task_ids": {
            "base_lost_by_sft": sorted(base_lost_by_sft),
            "sft_new_vs_base": sorted(base_new_in_sft),
            "base_lost_then_recovered_by_grpo": sorted(recovered_by_grpo),
            "sft_lost_by_grpo": sorted(sft_lost_by_grpo),
            "grpo_new_vs_sft": sorted(grpo_new_vs_sft),
        },
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze large-k capability support across Base/SFT/GRPO"
    )
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--sft", type=Path, required=True)
    p.add_argument("--grpo", type=Path, required=True)
    p.add_argument("--output", type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = compare_large_k(
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
