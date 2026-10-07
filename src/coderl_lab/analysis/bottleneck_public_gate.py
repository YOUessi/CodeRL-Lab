from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from coderl_lab.analysis.bottleneck_correctness_causality import (
    bootstrap_task_cluster,
    load_jsonl,
)


def load_reference_eval(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row["task_id"])
        if task_id in out:
            raise ValueError(f"duplicate reference task: {task_id}")
        out[task_id] = row
    return out


def summarize_gate(
    rows: list[dict[str, Any]],
    reference_eval: dict[str, dict[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    task_ids = sorted({str(row["task_id"]) for row in rows})
    task_info: dict[str, dict[str, Any]] = {}
    for task_id in task_ids:
        ref = reference_eval[task_id]
        public_rate = float(ref["public"]["pass_rate"])
        hidden_correct = bool(ref["hidden_all_pass"])
        task_info[task_id] = {
            "public_pass_rate": public_rate,
            "public_all_pass": public_rate == 1.0,
            "public_zero": public_rate == 0.0,
            "hidden_correct": hidden_correct,
        }

    confusion = {
        "public_pass_hidden_correct": 0,
        "public_pass_hidden_wrong": 0,
        "public_fail_hidden_correct": 0,
        "public_fail_hidden_wrong": 0,
    }
    for task_id, info in task_info.items():
        p = bool(info["public_all_pass"])
        h = bool(info["hidden_correct"])
        if p and h:
            confusion["public_pass_hidden_correct"] += 1
        elif p and not h:
            confusion["public_pass_hidden_wrong"] += 1
        elif not p and h:
            confusion["public_fail_hidden_correct"] += 1
        else:
            confusion["public_fail_hidden_wrong"] += 1

    public_fail_tasks = [
        task_id for task_id, info in task_info.items()
        if not bool(info["public_all_pass"])
    ]
    hidden_wrong_tasks = [
        task_id for task_id, info in task_info.items()
        if not bool(info["hidden_correct"])
    ]
    public_fail_hidden_wrong = confusion["public_fail_hidden_wrong"]

    gate_quality = {
        "eligible_tasks": len(task_ids),
        "public_fail_tasks": len(public_fail_tasks),
        "hidden_wrong_tasks": len(hidden_wrong_tasks),
        "public_fail_precision_for_hidden_wrong": (
            public_fail_hidden_wrong / len(public_fail_tasks)
            if public_fail_tasks else None
        ),
        "public_fail_recall_of_hidden_wrong": (
            public_fail_hidden_wrong / len(hidden_wrong_tasks)
            if hidden_wrong_tasks else None
        ),
        "confusion": confusion,
    }

    strata: dict[str, Any] = {}
    for stratum_name, predicate in (
        ("public_fail", lambda info: not bool(info["public_all_pass"])),
        ("public_pass", lambda info: bool(info["public_all_pass"])),
        ("public_zero", lambda info: bool(info["public_zero"])),
    ):
        task_set = {
            task_id for task_id, info in task_info.items()
            if predicate(info)
        }
        subset = [
            row for row in rows
            if str(row["task_id"]) in task_set
        ]
        if not subset:
            strata[stratum_name] = {
                "tasks": 0,
                "pairs": 0,
            }
            continue

        baseline = mean(
            1.0 if row["baseline_correct"] else 0.0 for row in subset
        )
        destabilize = mean(
            1.0 if row["low_destabilize_correct"] else 0.0
            for row in subset
        )
        stabilize = mean(
            1.0 if row["low_stabilize_correct"] else 0.0
            for row in subset
        )

        transitions = {
            "wrong_to_correct": sum(
                (not bool(row["baseline_correct"]))
                and bool(row["low_destabilize_correct"])
                for row in subset
            ),
            "correct_to_wrong": sum(
                bool(row["baseline_correct"])
                and (not bool(row["low_destabilize_correct"]))
                for row in subset
            ),
        }

        strata[stratum_name] = {
            "tasks": len(task_set),
            "pairs": len(subset),
            "reference_hidden_correct_tasks": sum(
                bool(task_info[t]["hidden_correct"]) for t in task_set
            ),
            "baseline_correct_fraction": baseline,
            "low_stabilize_correct_fraction": stabilize,
            "low_destabilize_correct_fraction": destabilize,
            "low_destabilize_delta": destabilize - baseline,
            "low_destabilize_transitions": transitions,
            "low_destabilize_minus_baseline": bootstrap_task_cluster(
                subset,
                value_fn=lambda row: (
                    float(bool(row["low_destabilize_correct"]))
                    - float(bool(row["baseline_correct"]))
                ),
                iterations=iterations,
                seed=seed,
            ),
            "low_stabilize_minus_baseline": bootstrap_task_cluster(
                subset,
                value_fn=lambda row: (
                    float(bool(row["low_stabilize_correct"]))
                    - float(bool(row["baseline_correct"]))
                ),
                iterations=iterations,
                seed=seed,
            ),
        }

    return {
        "gate_quality": gate_quality,
        "strata": strata,
        "task_info": task_info,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit public-test gating for bottleneck intervention"
    )
    parser.add_argument("--correctness-details", type=Path, required=True)
    parser.add_argument("--reference-details", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.correctness_details)
    reference_eval = load_reference_eval(args.reference_details)
    summary = summarize_gate(
        rows,
        reference_eval,
        iterations=args.iterations,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
