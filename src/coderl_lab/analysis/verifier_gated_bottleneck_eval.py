from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Any

from coderl_lab.analysis.bottleneck_correctness_causality import (
    bootstrap_task_cluster,
)
from coderl_lab.analysis.local_bottleneck_intervention import load_jsonl
from coderl_lab.evaluation import load_tasks
from coderl_lab.execution import PythonExecutor, report_to_dict
from coderl_lab.generation import extract_python_code


CONDITIONS = (
    "baseline",
    "gated_low_destabilize",
    "gated_high_destabilize",
    "always_low_destabilize",
)


def load_details(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row["task_id"])
        if task_id in out:
            raise ValueError(f"duplicate detail task: {task_id}")
        out[task_id] = row
    return out


def _evaluate_code(
    *,
    task,
    code: str,
    executor: PythonExecutor,
) -> dict[str, Any]:
    public = executor.run(
        code,
        entry_point=task.entry_point,
        cases=task.public_tests,
        setup_code=task.setup_code,
    )
    hidden = executor.run(
        code,
        entry_point=task.entry_point,
        cases=task.hidden_tests,
        setup_code=task.setup_code,
    )
    return {
        "public": report_to_dict(public),
        "hidden": report_to_dict(hidden),
        "hidden_all_pass": bool(hidden.syntax_ok and hidden.pass_rate == 1.0),
    }


def _transition_counts(
    rows: list[dict[str, Any]],
    condition: str,
) -> dict[str, int]:
    out = {
        "wrong_to_correct": 0,
        "correct_to_wrong": 0,
        "correct_to_correct": 0,
        "wrong_to_wrong": 0,
    }
    for row in rows:
        baseline = bool(row["baseline_correct"])
        candidate = bool(row[f"{condition}_correct"])
        if not baseline and candidate:
            out["wrong_to_correct"] += 1
        elif baseline and not candidate:
            out["correct_to_wrong"] += 1
        elif baseline and candidate:
            out["correct_to_correct"] += 1
        else:
            out["wrong_to_wrong"] += 1
    return out


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("no rows")

    result: dict[str, Any] = {
        "tasks": len(rows),
        "gate_triggered_tasks": sum(bool(row["gate_triggered"]) for row in rows),
        "eligible_tasks": sum(bool(row["eligible"]) for row in rows),
        "conditions": {},
        "contrasts": {},
    }

    for condition in CONDITIONS:
        correct = mean(
            1.0 if row[f"{condition}_correct"] else 0.0
            for row in rows
        )
        public_all = mean(
            1.0 if row[f"{condition}_public_all_pass"] else 0.0
            for row in rows
        )
        result["conditions"][condition] = {
            "hidden_correct_fraction": correct,
            "hidden_correct_tasks": sum(
                bool(row[f"{condition}_correct"]) for row in rows
            ),
            "public_all_pass_fraction": public_all,
            "public_all_pass_tasks": sum(
                bool(row[f"{condition}_public_all_pass"]) for row in rows
            ),
            "transitions_vs_baseline": (
                None
                if condition == "baseline"
                else _transition_counts(rows, condition)
            ),
        }

    baseline_fraction = result["conditions"]["baseline"][
        "hidden_correct_fraction"
    ]
    for condition in CONDITIONS[1:]:
        result["conditions"][condition]["delta_vs_baseline"] = (
            result["conditions"][condition]["hidden_correct_fraction"]
            - baseline_fraction
        )
        result["contrasts"][f"{condition}_minus_baseline"] = (
            bootstrap_task_cluster(
                rows,
                value_fn=lambda row, c=condition: (
                    float(bool(row[f"{c}_correct"]))
                    - float(bool(row["baseline_correct"]))
                ),
                iterations=iterations,
                seed=seed,
            )
        )

    for left, right in (
        ("gated_low_destabilize", "gated_high_destabilize"),
        ("gated_low_destabilize", "always_low_destabilize"),
    ):
        result["contrasts"][f"{left}_minus_{right}"] = (
            bootstrap_task_cluster(
                rows,
                value_fn=lambda row, a=left, b=right: (
                    float(bool(row[f"{a}_correct"]))
                    - float(bool(row[f"{b}_correct"]))
                ),
                iterations=iterations,
                seed=seed,
            )
        )

    gated = [row for row in rows if bool(row["gate_triggered"])]
    if gated:
        result["gate_triggered_subset"] = {
            "tasks": len(gated),
            "baseline_correct_fraction": mean(
                1.0 if row["baseline_correct"] else 0.0
                for row in gated
            ),
            "gated_low_correct_fraction": mean(
                1.0 if row["gated_low_destabilize_correct"] else 0.0
                for row in gated
            ),
            "gated_high_correct_fraction": mean(
                1.0 if row["gated_high_destabilize_correct"] else 0.0
                for row in gated
            ),
            "low_transitions": _transition_counts(
                gated,
                "gated_low_destabilize",
            ),
            "high_transitions": _transition_counts(
                gated,
                "gated_high_destabilize",
            ),
            "low_minus_baseline": bootstrap_task_cluster(
                gated,
                value_fn=lambda row: (
                    float(bool(row["gated_low_destabilize_correct"]))
                    - float(bool(row["baseline_correct"]))
                ),
                iterations=iterations,
                seed=seed,
            ),
            "low_minus_high": bootstrap_task_cluster(
                gated,
                value_fn=lambda row: (
                    float(bool(row["gated_low_destabilize_correct"]))
                    - float(bool(row["gated_high_destabilize_correct"]))
                ),
                iterations=iterations,
                seed=seed,
            ),
        }

    public_pass = [
        row for row in rows
        if bool(row["baseline_public_all_pass"])
    ]
    result["baseline_public_pass_subset"] = {
        "tasks": len(public_pass),
        "gated_low_changed": sum(
            str(row["gated_low_destabilize_raw"])
            != str(row["baseline_raw"])
            for row in public_pass
        ),
        "gated_high_changed": sum(
            str(row["gated_high_destabilize_raw"])
            != str(row["baseline_raw"])
            for row in public_pass
        ),
        "always_low_changed": sum(
            str(row["always_low_destabilize_raw"])
            != str(row["baseline_raw"])
            for row in public_pass
        ),
    }

    return result


def run_evaluation(
    *,
    runner_path: Path,
    tasks_path: Path,
    reference_details_path: Path,
    output_dir: Path,
    workers: int,
    timeout: float,
    memory: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    runner = json.loads(runner_path.read_text(encoding="utf-8"))
    tasks = load_tasks(tasks_path)
    baseline_details = load_details(reference_details_path)

    executor = PythonExecutor(
        mode="docker",
        timeout_seconds=timeout,
        memory_limit=memory,
    )

    rows: list[dict[str, Any]] = []
    jobs: dict[tuple[str, str], tuple[str, str]] = {}

    for task_id, item in runner["per_task"].items():
        task = tasks[task_id]
        baseline_raw = str(item["baseline"]["raw_completion"])
        row: dict[str, Any] = {
            "task_id": task_id,
            "eligible": bool(item["eligible"]),
            "gate_triggered": bool(item["gate_triggered"]),
            "baseline_raw": baseline_raw,
            "baseline_correct": bool(
                baseline_details[task_id]["hidden_all_pass"]
            ),
            "baseline_public_all_pass": (
                float(baseline_details[task_id]["public"]["pass_rate"]) == 1.0
            ),
        }

        for condition in CONDITIONS:
            raw = str(item[condition]["raw_completion"])
            row[f"{condition}_raw"] = raw
            row[f"{condition}_intervention_applied"] = bool(
                item[condition].get("intervention_applied", False)
            )
            if condition == "baseline" or raw == baseline_raw:
                continue
            code = extract_python_code(raw, task.entry_point)
            jobs[(task_id, code)] = (task_id, code)

        rows.append(row)

    def worker(job):
        key, (task_id, code) = job
        return key, _evaluate_code(
            task=tasks[task_id],
            code=code,
            executor=executor,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        evaluated = dict(pool.map(worker, jobs.items()))

    detail_rows: list[dict[str, Any]] = []
    for row in rows:
        task_id = str(row["task_id"])
        task = tasks[task_id]
        baseline_detail = baseline_details[task_id]

        for condition in CONDITIONS:
            raw = str(row[f"{condition}_raw"])
            if condition == "baseline" or raw == row["baseline_raw"]:
                detail = baseline_detail
            else:
                code = extract_python_code(raw, task.entry_point)
                detail = evaluated[(task_id, code)]

            row[f"{condition}_correct"] = bool(detail["hidden_all_pass"])
            row[f"{condition}_public_all_pass"] = (
                float(detail["public"]["pass_rate"]) == 1.0
            )
            row[f"{condition}_hidden_pass_rate"] = float(
                detail["hidden"]["pass_rate"]
            )

        detail_rows.append(row)

    summary = summarize_rows(
        detail_rows,
        iterations=iterations,
        seed=seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "details.jsonl").open("w", encoding="utf-8") as handle:
        for row in detail_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--reference-details", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--memory", default="512m")
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluation(
        runner_path=args.runner,
        tasks_path=args.tasks,
        reference_details_path=args.reference_details,
        output_dir=args.output,
        workers=args.workers,
        timeout=args.timeout,
        memory=args.memory,
        iterations=args.iterations,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
