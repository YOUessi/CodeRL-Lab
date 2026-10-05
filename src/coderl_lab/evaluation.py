from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .execution import PythonExecutor, report_to_dict
from .metrics import mean_pass_at_k
from .reward import compute_training_reward
from .schema import CodeTask


def load_tasks(path: Path) -> dict[str, CodeTask]:
    tasks: dict[str, CodeTask] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            task = CodeTask.from_dict(data)
            if task.task_id in tasks:
                raise ValueError(f"duplicate task_id at line {line_no}: {task.task_id}")
            tasks[task.task_id] = task
    if not tasks:
        raise ValueError("task file is empty")
    return tasks


def load_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "task_id" not in row or "completion" not in row:
                raise ValueError(
                    f"prediction line {line_no} requires task_id and completion"
                )
            sample_id = int(row.get("sample_id", 0))
            key = (str(row["task_id"]), sample_id)
            if key in seen:
                raise ValueError(f"duplicate prediction key: {key}")
            seen.add(key)
            row["sample_id"] = sample_id
            rows.append(row)
    if not rows:
        raise ValueError("prediction file is empty")
    return rows


def evaluate_predictions(
    *,
    tasks: dict[str, CodeTask],
    predictions: list[dict[str, Any]],
    executor: PythonExecutor,
    ks: tuple[int, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    detailed: list[dict[str, Any]] = []
    hidden_counts: dict[str, list[bool]] = defaultdict(list)

    for row in predictions:
        task_id = str(row["task_id"])
        if task_id not in tasks:
            raise KeyError(f"prediction refers to unknown task_id: {task_id}")
        task = tasks[task_id]
        completion = str(row["completion"])

        public_report = executor.run(
            completion,
            task_mode=task.task_mode,
            entry_point=task.entry_point,
            cases=task.public_tests,
        )
        hidden_report = executor.run(
            completion,
            task_mode=task.task_mode,
            entry_point=task.entry_point,
            cases=task.hidden_tests,
        )

        training_reward = compute_training_reward(
            syntax_ok=public_report.syntax_ok,
            public_pass_rate=public_report.pass_rate,
        )
        hidden_correct = hidden_report.syntax_ok and hidden_report.pass_rate == 1.0
        hidden_counts[task_id].append(hidden_correct)

        detailed.append(
            {
                "task_id": task_id,
                "task_mode": task.task_mode,
                "sample_id": int(row["sample_id"]),
                "training_reward": {
                    "syntax": training_reward.syntax,
                    "public_pass_rate": training_reward.public_pass_rate,
                    "all_public_pass_bonus": training_reward.all_public_pass_bonus,
                    "total": training_reward.total,
                },
                "public": report_to_dict(public_report),
                "hidden": report_to_dict(hidden_report),
                "hidden_all_pass": hidden_correct,
            }
        )

    pass_at_k: dict[str, float] = {}
    task_counts = [
        (len(outcomes), sum(outcomes))
        for outcomes in hidden_counts.values()
    ]
    for k in ks:
        eligible = [(n, c) for n, c in task_counts if n >= k]
        if eligible:
            pass_at_k[f"pass@{k}"] = mean_pass_at_k(eligible, k)

    summary = {
        "num_tasks": len(hidden_counts),
        "num_predictions": len(predictions),
        "pass_at_k": pass_at_k,
        "per_task": {
            task_id: {
                "samples": len(outcomes),
                "correct": sum(outcomes),
                "empirical_success_rate": sum(outcomes) / len(outcomes),
            }
            for task_id, outcomes in sorted(hidden_counts.items())
        },
    }
    return detailed, summary


def write_outputs(
    output_dir: Path,
    detailed: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "details.jsonl"
    with details_path.open("w", encoding="utf-8") as handle:
        for row in detailed:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CodeRL-Lab predictions")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executor", choices=["docker", "local"], default="docker")
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--memory", default="512m")
    parser.add_argument(
        "--allow-unsafe-local",
        action="store_true",
        help="Only for trusted fixtures. Never use for model-generated code.",
    )
    parser.add_argument("--k", type=int, nargs="+", default=[1, 4, 8, 16])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_tasks(args.tasks)
    predictions = load_predictions(args.predictions)
    executor = PythonExecutor(
        mode=args.executor,
        timeout_seconds=args.timeout,
        memory_limit=args.memory,
        allow_unsafe_local=args.allow_unsafe_local,
    )
    details, summary = evaluate_predictions(
        tasks=tasks,
        predictions=predictions,
        executor=executor,
        ks=tuple(args.k),
    )
    write_outputs(args.output, details, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
