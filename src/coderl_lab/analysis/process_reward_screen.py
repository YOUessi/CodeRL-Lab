from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from coderl_lab.execution import PythonExecutor
from coderl_lab.generation import extract_python_code
from coderl_lab.reward import (
    analyze_dependency_completeness,
    compute_execution_stage_reward,
    compute_runtime_clean_rate,
    compute_training_reward,
)
from coderl_lab.schema import TestCase


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _score_one(
    row: dict[str, Any],
    *,
    task_map: dict[str, dict[str, Any]],
    executor: PythonExecutor,
) -> dict[str, Any]:
    task_id = str(row["task_id"])
    task = task_map[task_id]
    entry_point = str(task["entry_point"])
    setup_code = str(task.get("setup_code", ""))
    code = extract_python_code(str(row["completion"]), entry_point)

    dep = analyze_dependency_completeness(
        code,
        entry_point=entry_point,
        setup_code=setup_code,
    )
    report = executor.run(
        code,
        entry_point=entry_point,
        cases=tuple(TestCase.from_dict(x) for x in task["public_tests"]),
        setup_code=setup_code,
    )
    runtime_clean = compute_runtime_clean_rate(report.cases)

    outcome = compute_training_reward(
        syntax_ok=report.syntax_ok,
        public_pass_rate=report.pass_rate,
    )
    process = compute_execution_stage_reward(
        syntax_ok=report.syntax_ok,
        dependency_complete=dep.complete,
        runtime_clean_rate=runtime_clean,
        public_pass_rate=report.pass_rate,
    )

    return {
        "task_id": task_id,
        "sample_id": int(row["sample_id"]),
        "syntax_ok": bool(report.syntax_ok),
        "dependency_complete": bool(dep.complete),
        "unresolved_names": list(dep.unresolved_names),
        "runtime_clean_rate": float(runtime_clean),
        "public_pass_rate": float(report.pass_rate),
        "outcome_reward": float(outcome.total),
        "process_reward": float(process.total),
        "process_components": {
            "syntax": float(process.syntax),
            "dependency_complete": float(process.dependency_complete),
            "runtime_clean_rate": float(process.runtime_clean_rate),
            "public_pass_rate": float(process.public_pass_rate),
            "all_public_pass_bonus": float(process.all_public_pass_bonus),
        },
    }


def _is_flat(values: list[float], eps: float = 1e-12) -> bool:
    return max(values) - min(values) <= eps


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_id"])].append(row)

    per_task: dict[str, Any] = {}
    outcome_flat = 0
    process_flat = 0
    rescued = 0
    collapsed = 0
    rescued_zero_public = 0

    for task_id, task_rows in sorted(grouped.items()):
        task_rows.sort(key=lambda x: int(x["sample_id"]))
        old = [float(x["outcome_reward"]) for x in task_rows]
        new = [float(x["process_reward"]) for x in task_rows]
        old_flat = _is_flat(old)
        new_flat = _is_flat(new)

        if old_flat:
            outcome_flat += 1
        if new_flat:
            process_flat += 1
        if old_flat and not new_flat:
            rescued += 1
            if all(float(x["public_pass_rate"]) == 0.0 for x in task_rows):
                rescued_zero_public += 1
        if not old_flat and new_flat:
            collapsed += 1

        per_task[task_id] = {
            "samples": len(task_rows),
            "outcome_reward_mean": sum(old) / len(old),
            "outcome_reward_spread": max(old) - min(old),
            "process_reward_mean": sum(new) / len(new),
            "process_reward_spread": max(new) - min(new),
            "outcome_flat": old_flat,
            "process_flat": new_flat,
            "dependency_complete_rate": (
                sum(bool(x["dependency_complete"]) for x in task_rows)
                / len(task_rows)
            ),
            "runtime_clean_mean": (
                sum(float(x["runtime_clean_rate"]) for x in task_rows)
                / len(task_rows)
            ),
            "public_pass_mean": (
                sum(float(x["public_pass_rate"]) for x in task_rows)
                / len(task_rows)
            ),
        }

    n_tasks = len(per_task)
    n_rows = len(rows)
    return {
        "num_tasks": n_tasks,
        "num_predictions": n_rows,
        "outcome_flat_tasks": outcome_flat,
        "outcome_flat_fraction": outcome_flat / n_tasks,
        "process_flat_tasks": process_flat,
        "process_flat_fraction": process_flat / n_tasks,
        "flat_tasks_rescued_by_process_reward": rescued,
        "flat_rescue_fraction_of_outcome_flat": (
            rescued / outcome_flat if outcome_flat else 0.0
        ),
        "rescued_zero_public_tasks": rescued_zero_public,
        "mixed_tasks_collapsed_by_process_reward": collapsed,
        "overall_dependency_complete_rate": (
            sum(bool(x["dependency_complete"]) for x in rows) / n_rows
        ),
        "overall_runtime_clean_mean": (
            sum(float(x["runtime_clean_rate"]) for x in rows) / n_rows
        ),
        "overall_public_pass_mean": (
            sum(float(x["public_pass_rate"]) for x in rows) / n_rows
        ),
        "overall_outcome_reward_mean": (
            sum(float(x["outcome_reward"]) for x in rows) / n_rows
        ),
        "overall_process_reward_mean": (
            sum(float(x["process_reward"]) for x in rows) / n_rows
        ),
        "per_task": per_task,
    }


def run(
    *,
    tasks_path: Path,
    predictions_path: Path,
    output_dir: Path,
    workers: int,
    docker_image: str,
    timeout_seconds: float,
    memory_limit: str,
) -> dict[str, Any]:
    tasks = load_jsonl(tasks_path)
    predictions = load_jsonl(predictions_path)
    task_map = {str(row["task_id"]): row for row in tasks}

    unknown = sorted(
        {str(row["task_id"]) for row in predictions} - set(task_map)
    )
    if unknown:
        raise ValueError(f"predictions contain unknown tasks: {unknown[:10]}")

    executor = PythonExecutor(
        mode="docker",
        timeout_seconds=timeout_seconds,
        docker_image=docker_image,
        memory_limit=memory_limit,
    )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(
            pool.map(
                lambda row: _score_one(
                    row,
                    task_map=task_map,
                    executor=executor,
                ),
                predictions,
            )
        )

    rows.sort(key=lambda x: (x["task_id"], int(x["sample_id"])))
    summary = summarize(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "details.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare outcome-only and execution-stage reward variance"
    )
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--docker-image", default="python:3.11-slim")
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--memory", default="512m")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(
        tasks_path=args.tasks,
        predictions_path=args.predictions,
        output_dir=args.output,
        workers=args.workers,
        docker_image=args.docker_image,
        timeout_seconds=args.timeout,
        memory_limit=args.memory,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
