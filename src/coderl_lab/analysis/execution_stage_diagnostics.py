from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from coderl_lab.reward import (
    analyze_dependency_completeness,
    compute_runtime_clean_rate,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _error_type(error: str | None) -> str:
    if not error:
        return "none"
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    if not lines:
        return "unknown"
    last = lines[-1]
    if ":" in last:
        return last.split(":", 1)[0]
    return last


def diagnose(
    *,
    tasks: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    details: list[dict[str, Any]],
) -> dict[str, Any]:
    task_map = {str(row["task_id"]): row for row in tasks}
    prediction_map = {
        (str(row["task_id"]), int(row["sample_id"])): row
        for row in predictions
    }

    rows: list[dict[str, Any]] = []
    unresolved_counter: Counter[str] = Counter()
    runtime_failure_types: Counter[str] = Counter()

    for detail in details:
        key = (str(detail["task_id"]), int(detail["sample_id"]))
        pred = prediction_map.get(key)
        if pred is None:
            raise ValueError(f"prediction missing for detail row: {key}")
        task = task_map.get(key[0])
        if task is None:
            raise ValueError(f"task missing for detail row: {key[0]}")

        dep = analyze_dependency_completeness(
            str(pred["completion"]),
            entry_point=str(task["entry_point"]),
            setup_code=str(task.get("setup_code", "")),
        )
        public_cases = list(detail["public"]["cases"])
        runtime_clean_rate = compute_runtime_clean_rate(public_cases)

        for name in dep.unresolved_names:
            unresolved_counter[name] += 1

        for case in public_cases:
            if not (
                (case.get("error") is None)
                or ("AssertionError" in str(case.get("error")))
            ):
                runtime_failure_types[_error_type(case.get("error"))] += 1

        rows.append(
            {
                "task_id": key[0],
                "sample_id": key[1],
                "dependency_complete": dep.complete,
                "unresolved_names": list(dep.unresolved_names),
                "runtime_clean_rate": runtime_clean_rate,
                "public_pass_rate": float(detail["public"]["pass_rate"]),
                "hidden_all_pass": bool(detail["hidden_all_pass"]),
                "syntax_ok": bool(detail["public"]["syntax_ok"]),
            }
        )

    def subset_summary(subset: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(subset)
        if n == 0:
            return {
                "count": 0,
                "dependency_complete_rate": None,
                "runtime_clean_mean": None,
                "syntax_ok_rate": None,
                "public_pass_mean": None,
            }
        return {
            "count": n,
            "dependency_complete_rate": (
                sum(bool(x["dependency_complete"]) for x in subset) / n
            ),
            "runtime_clean_mean": (
                sum(float(x["runtime_clean_rate"]) for x in subset) / n
            ),
            "syntax_ok_rate": (
                sum(bool(x["syntax_ok"]) for x in subset) / n
            ),
            "public_pass_mean": (
                sum(float(x["public_pass_rate"]) for x in subset) / n
            ),
        }

    correct = [row for row in rows if row["hidden_all_pass"]]
    incorrect = [row for row in rows if not row["hidden_all_pass"]]
    dep_incomplete = [row for row in rows if not row["dependency_complete"]]
    runtime_unclean = [row for row in rows if row["runtime_clean_rate"] < 1.0]

    return {
        "num_predictions": len(rows),
        "overall": subset_summary(rows),
        "hidden_correct": subset_summary(correct),
        "hidden_incorrect": subset_summary(incorrect),
        "dependency_incomplete_count": len(dep_incomplete),
        "runtime_unclean_count": len(runtime_unclean),
        "dependency_incomplete_and_hidden_incorrect": sum(
            not row["dependency_complete"] and not row["hidden_all_pass"]
            for row in rows
        ),
        "runtime_unclean_and_hidden_incorrect": sum(
            row["runtime_clean_rate"] < 1.0 and not row["hidden_all_pass"]
            for row in rows
        ),
        "top_unresolved_names": unresolved_counter.most_common(25),
        "runtime_failure_types": runtime_failure_types.most_common(25),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnose execution-stage signals in fixed CodeRL-Lab outputs"
    )
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--details", type=Path, required=True)
    p.add_argument("--output", type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = diagnose(
        tasks=load_jsonl(args.tasks),
        predictions=load_jsonl(args.predictions),
        details=load_jsonl(args.details),
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
