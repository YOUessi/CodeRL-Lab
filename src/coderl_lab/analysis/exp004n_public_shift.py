"""Outcome-blind comparison of pinned v5 and v6 public task distributions.

Only metadata and prompt-length proxies are used. No model predictions,
private tests or correctness labels enter this diagnostic.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from coderl_lab.analysis.livecodebench_official_eval import load_public_tasks
from coderl_lab.datasets.livecodebench import sha256_file

V5_PUBLIC = "e695ba9fa2ce35abc8db2f3360bf711930746cd55843890177ecd518c0a4c98d"
V6_PUBLIC = "f9fd88d4e1b35b4f6720ca548c2e2d1187ad53b5fb5723ef4999a40b79d7c399"


def summarize_tasks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("no tasks")
    platforms = Counter(str(x["platform"]).lower() for x in rows)
    difficulties = Counter(str(x["difficulty"]).lower() for x in rows)
    tests = [len(x["public_test_cases"]) for x in rows]
    content_chars = [len(str(x["question_content"])) for x in rows]
    starter_chars = [len(str(x.get("starter_code", ""))) for x in rows]
    by_difficulty: dict[str, Any] = {}
    for difficulty in sorted(difficulties):
        chosen = [x for x in rows if str(x["difficulty"]).lower() == difficulty]
        by_difficulty[difficulty] = {
            "tasks": len(chosen),
            "question_chars_mean": mean(len(str(x["question_content"])) for x in chosen),
            "with_starter": sum(bool(str(x.get("starter_code", "")).strip()) for x in chosen),
            "public_tests": sum(len(x["public_test_cases"]) for x in chosen),
        }
    return {
        "tasks": len(rows),
        "platform_counts": dict(sorted(platforms.items())),
        "difficulty_counts": dict(sorted(difficulties.items())),
        "public_test_cases": sum(tests),
        "public_tests_per_task_mean": mean(tests),
        "question_chars_mean": mean(content_chars),
        "question_chars_median": median(content_chars),
        "question_chars_max": max(content_chars),
        "starter_chars_mean": mean(starter_chars),
        "with_starter_count": sum(bool(v) for v in starter_chars),
        "by_difficulty": by_difficulty,
        "earliest_contest_date": min(str(x["contest_date"]) for x in rows),
        "latest_contest_date": max(str(x["contest_date"]) for x in rows),
    }


def compare_v5_v6(
    *, v5_public: Path, v6_public: Path,
    require_pinned: bool = True,
) -> dict[str, Any]:
    if require_pinned:
        if sha256_file(v5_public) != V5_PUBLIC:
            raise ValueError("v5 source is not pinned public-only view")
        if sha256_file(v6_public) != V6_PUBLIC:
            raise ValueError("v6 source is not pinned public-only view")
    v5 = load_public_tasks(v5_public)
    v6 = load_public_tasks(v6_public)
    ids_overlap = set(v5) & set(v6)
    if ids_overlap:
        raise ValueError("v5/v6 task ID overlap detected")
    if require_pinned and (len(v5) != 167 or len(v6) != 175):
        raise ValueError("v5/v6 expected count mismatch")
    fingerprint = lambda row: " ".join(
        (str(row["question_content"]) + "\n" + str(row.get("starter_code", "")))
        .casefold().split()
    )
    fp5 = {fingerprint(v) for v in v5.values()}
    fp6 = {fingerprint(v) for v in v6.values()}
    if fp5 & fp6:
        raise ValueError("same normalized v5/v6 task texts")
    a = summarize_tasks(list(v5.values()))
    b = summarize_tasks(list(v6.values()))
    difficulties = sorted(set(a["difficulty_counts"]) | set(b["difficulty_counts"]))
    return {
        "experiment": "EXP-004N",
        "stage": "public_distribution_before_model_outcomes",
        "uses_private_tests": False,
        "uses_generated_answers": False,
        "v5_public_sha256": sha256_file(v5_public),
        "v6_public_sha256": sha256_file(v6_public),
        "overlap_task_id": 0, "overlap_normalized_problem": 0,
        "v5": a, "v6": b,
        "v6_minus_v5_difficulty_fraction": {
            d: b["difficulty_counts"].get(d, 0) / b["tasks"] -
               a["difficulty_counts"].get(d, 0) / a["tasks"]
            for d in difficulties
        },
        "interpretation_limit": (
            "Observable task composition only. Prompt characters/difficulty are "
            "not correctness, ability or contaminated-model measures."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--v5", type=Path, default=Path("artifacts/exp004n/data/public_tasks.jsonl"))
    p.add_argument("--v6", type=Path, default=Path("artifacts/exp004l/data/public_tasks.jsonl"))
    p.add_argument("--output", type=Path, default=Path("artifacts/exp004n/public_shift/summary.json"))
    args = p.parse_args()
    result = compare_v5_v6(v5_public=args.v5, v6_public=args.v6)
    if args.output.exists():
        raise FileExistsError("frozen public-distribution audit already saved")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
