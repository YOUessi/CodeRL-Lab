from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from coderl_lab.analysis.bottleneck_correctness_causality import (
    bootstrap_task_cluster,
    load_jsonl,
)
from coderl_lab.analysis.bottleneck_public_gate import load_reference_eval


def summarize_gated_policy(
    rows: list[dict[str, Any]],
    reference_eval: dict[str, dict[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        task_id = str(row["task_id"])
        public_fail = float(reference_eval[task_id]["public"]["pass_rate"]) < 1.0
        baseline = bool(row["baseline_correct"])
        chosen = (
            bool(row["low_destabilize_correct"])
            if public_fail
            else baseline
        )
        enriched.append(
            {
                **row,
                "gate_public_fail": public_fail,
                "gated_correct": chosen,
            }
        )

    baseline_fraction = mean(
        1.0 if row["baseline_correct"] else 0.0
        for row in enriched
    )
    gated_fraction = mean(
        1.0 if row["gated_correct"] else 0.0
        for row in enriched
    )

    transitions = {
        "wrong_to_correct": sum(
            (not bool(row["baseline_correct"]))
            and bool(row["gated_correct"])
            for row in enriched
        ),
        "correct_to_wrong": sum(
            bool(row["baseline_correct"])
            and (not bool(row["gated_correct"]))
            for row in enriched
        ),
        "correct_to_correct": sum(
            bool(row["baseline_correct"])
            and bool(row["gated_correct"])
            for row in enriched
        ),
        "wrong_to_wrong": sum(
            (not bool(row["baseline_correct"]))
            and (not bool(row["gated_correct"]))
            for row in enriched
        ),
    }

    return {
        "eligible_tasks": len({str(row["task_id"]) for row in enriched}),
        "pairs": len(enriched),
        "public_fail_pairs": sum(
            bool(row["gate_public_fail"]) for row in enriched
        ),
        "baseline_correct_fraction": baseline_fraction,
        "gated_correct_fraction": gated_fraction,
        "delta": gated_fraction - baseline_fraction,
        "transitions": transitions,
        "task_cluster_bootstrap": bootstrap_task_cluster(
            enriched,
            value_fn=lambda row: (
                float(bool(row["gated_correct"]))
                - float(bool(row["baseline_correct"]))
            ),
            iterations=iterations,
            seed=seed,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a public-test-gated bottleneck policy"
    )
    parser.add_argument("--correctness-details", type=Path, required=True)
    parser.add_argument("--reference-details", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parser_args = parse_args()
    rows = load_jsonl(parser_args.correctness_details)
    reference_eval = load_reference_eval(parser_args.reference_details)
    summary = summarize_gated_policy(
        rows,
        reference_eval,
        iterations=parser_args.iterations,
        seed=parser_args.seed,
    )
    parser_args.output.parent.mkdir(parents=True, exist_ok=True)
    parser_args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
