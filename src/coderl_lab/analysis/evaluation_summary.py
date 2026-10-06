from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize_evaluation(
    details: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    if not details:
        raise ValueError("evaluation details are empty")

    num_predictions = len(details)
    syntax_failures = sum(
        not bool(row["public"]["syntax_ok"]) for row in details
    )
    public_mean = sum(
        float(row["public"]["pass_rate"]) for row in details
    ) / num_predictions
    hidden_mean = sum(
        float(row["hidden"]["pass_rate"]) for row in details
    ) / num_predictions
    public_all = sum(
        float(row["public"]["pass_rate"]) == 1.0 for row in details
    )
    hidden_all = sum(bool(row["hidden_all_pass"]) for row in details)
    public_all_hidden_fail = sum(
        float(row["public"]["pass_rate"]) == 1.0
        and not bool(row["hidden_all_pass"])
        for row in details
    )

    per_task = dict(summary["per_task"])
    solved_tasks = sum(
        int(row["correct"]) > 0 for row in per_task.values()
    )
    all_samples_correct_tasks = sum(
        int(row["correct"]) == int(row["samples"])
        for row in per_task.values()
    )

    return {
        "num_tasks": int(summary["num_tasks"]),
        "num_predictions": num_predictions,
        "pass_at_k": dict(summary["pass_at_k"]),
        "syntax_failures": syntax_failures,
        "syntax_failure_rate": syntax_failures / num_predictions,
        "public_mean_pass_rate": public_mean,
        "hidden_mean_pass_rate": hidden_mean,
        "public_all_pass_candidates": public_all,
        "hidden_all_pass_candidates": hidden_all,
        "public_all_hidden_fail_candidates": public_all_hidden_fail,
        "solved_tasks": solved_tasks,
        "all_samples_correct_tasks": all_samples_correct_tasks,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an enhanced CodeRL-Lab evaluation summary"
    )
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    details = load_jsonl(args.details)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    result = summarize_evaluation(details, summary)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
