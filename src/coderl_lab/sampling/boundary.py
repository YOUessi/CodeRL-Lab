from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import random
from statistics import mean
from typing import Any

from coderl_lab.execution import PythonExecutor
from coderl_lab.generation import extract_python_code
from coderl_lab.reward import compute_training_reward
from coderl_lab.schema import TestCase


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_reward_group(
    rewards: list[float],
    public_pass_rates: list[float],
    syntax_values: list[float],
    *,
    epsilon: float = 1e-9,
) -> str:
    if not rewards:
        raise ValueError("reward group cannot be empty")

    spread = max(rewards) - min(rewards)
    if spread > epsilon:
        return "mixed"

    if all(abs(x - 1.0) <= epsilon for x in rewards):
        return "flat_all_pass"
    if all(abs(x) <= epsilon for x in syntax_values):
        return "flat_syntax_fail"
    if all(abs(x) <= epsilon for x in public_pass_rates):
        return "flat_test_fail"
    return "flat_partial"


def select_random_control_ids(
    task_ids: list[str],
    count: int,
    *,
    seed: int,
) -> list[str]:
    if count < 0 or count > len(task_ids):
        raise ValueError("random control count is out of range")
    return random.Random(seed).sample(task_ids, count)


def strip_hidden_for_training(task: dict[str, Any]) -> dict[str, Any]:
    """Create a GRPO training row that structurally contains no hidden tests."""
    required = (
        "task_id",
        "prompt",
        "entry_point",
        "starter_code",
        "public_tests",
    )
    missing = [key for key in required if key not in task]
    if missing:
        raise ValueError(
            f"task {task.get('task_id')} missing fields: {', '.join(missing)}"
        )

    return {
        "task_id": task["task_id"],
        "prompt": task["prompt"],
        "entry_point": task["entry_point"],
        "starter_code": task.get("starter_code", ""),
        "public_tests": task["public_tests"],
        "setup_code": task.get("setup_code", ""),
        "metadata": {
            **dict(task.get("metadata", {})),
            "boundary_selected": True,
        },
    }


class PublicRewardScreener:
    """Score candidate completions using public tests only."""

    def __init__(
        self,
        *,
        timeout_seconds: float,
        memory_limit: str,
        docker_image: str,
        max_workers: int,
    ) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self.executor = PythonExecutor(
            mode="docker",
            timeout_seconds=timeout_seconds,
            docker_image=docker_image,
            memory_limit=memory_limit,
        )
        self.max_workers = max_workers

    def _score_one(
        self,
        task: dict[str, Any],
        prediction: dict[str, Any],
    ) -> dict[str, Any]:
        entry_point = str(task["entry_point"])
        raw = str(prediction["completion"])
        code = extract_python_code(raw, entry_point)
        public_tests = tuple(
            TestCase.from_dict(item)
            for item in task["public_tests"]
        )
        report = self.executor.run(
            code,
            entry_point=entry_point,
            cases=public_tests,
            setup_code=str(task.get("setup_code", "")),
        )
        reward = compute_training_reward(
            syntax_ok=report.syntax_ok,
            public_pass_rate=report.pass_rate,
        )
        return {
            "task_id": str(task["task_id"]),
            "sample_id": int(prediction["sample_id"]),
            "reward": float(reward.total),
            "syntax": float(reward.syntax),
            "public_pass_rate": float(reward.public_pass_rate),
            "all_public_pass_bonus": float(
                reward.all_public_pass_bonus
            ),
        }

    def score(
        self,
        tasks: dict[str, dict[str, Any]],
        predictions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        work: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for prediction in predictions:
            task_id = str(prediction["task_id"])
            if task_id not in tasks:
                raise KeyError(
                    f"prediction references unknown task: {task_id}"
                )
            work.append((tasks[task_id], prediction))

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(
                pool.map(
                    lambda item: self._score_one(*item),
                    work,
                )
            )


def build_screening(
    *,
    tasks_path: Path,
    predictions_path: Path,
    output_dir: Path,
    num_samples_per_task: int,
    timeout_seconds: float,
    memory_limit: str,
    docker_image: str,
    max_workers: int,
    random_control_seed: int = 42,
) -> dict[str, Any]:
    task_rows = _load_jsonl(tasks_path)
    prediction_rows = _load_jsonl(predictions_path)

    tasks = {str(row["task_id"]): row for row in task_rows}
    if len(tasks) != len(task_rows):
        raise ValueError("duplicate task_id in task file")

    screener = PublicRewardScreener(
        timeout_seconds=timeout_seconds,
        memory_limit=memory_limit,
        docker_image=docker_image,
        max_workers=max_workers,
    )
    scored = screener.score(tasks, prediction_rows)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        grouped[row["task_id"]].append(row)

    screening_rows: list[dict[str, Any]] = []
    selected_ids: list[str] = []

    for task_id in tasks:
        rows = sorted(
            grouped.get(task_id, []),
            key=lambda item: item["sample_id"],
        )
        if len(rows) != num_samples_per_task:
            raise ValueError(
                f"{task_id} has {len(rows)} samples; "
                f"expected {num_samples_per_task}"
            )

        rewards = [float(row["reward"]) for row in rows]
        public_pass_rates = [
            float(row["public_pass_rate"]) for row in rows
        ]
        syntax_values = [float(row["syntax"]) for row in rows]
        category = classify_reward_group(
            rewards,
            public_pass_rates,
            syntax_values,
        )

        if category == "mixed":
            selected_ids.append(task_id)

        screening_rows.append(
            {
                "task_id": task_id,
                "category": category,
                "rewards": rewards,
                "reward_mean": mean(rewards),
                "reward_min": min(rewards),
                "reward_max": max(rewards),
                "reward_spread": max(rewards) - min(rewards),
                "public_pass_rates": public_pass_rates,
                "syntax_values": syntax_values,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    screening_path = output_dir / "screening.jsonl"
    with screening_path.open("w", encoding="utf-8") as handle:
        for row in screening_rows:
            handle.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    selected_set = set(selected_ids)
    selected_path = output_dir / "boundary_train_tasks.jsonl"
    with selected_path.open("w", encoding="utf-8") as handle:
        for row in task_rows:
            if str(row["task_id"]) in selected_set:
                stripped = strip_hidden_for_training(row)
                handle.write(
                    json.dumps(stripped, ensure_ascii=False) + "\n"
                )

    all_task_ids = [str(row["task_id"]) for row in task_rows]
    random_control_ids = select_random_control_ids(
        all_task_ids,
        len(selected_ids),
        seed=random_control_seed,
    )
    random_control_set = set(random_control_ids)
    random_control_path = output_dir / "random_control_train_tasks.jsonl"
    with random_control_path.open("w", encoding="utf-8") as handle:
        for row in task_rows:
            if str(row["task_id"]) in random_control_set:
                stripped = strip_hidden_for_training(row)
                stripped["metadata"]["random_control"] = True
                stripped["metadata"]["random_control_seed"] = random_control_seed
                handle.write(
                    json.dumps(stripped, ensure_ascii=False) + "\n"
                )

    category_counts = Counter(
        row["category"] for row in screening_rows
    )
    summary = {
        "tasks_path": str(tasks_path),
        "tasks_sha256": _sha256(tasks_path),
        "predictions_path": str(predictions_path),
        "predictions_sha256": _sha256(predictions_path),
        "num_tasks": len(task_rows),
        "num_predictions": len(prediction_rows),
        "num_samples_per_task": num_samples_per_task,
        "category_counts": dict(sorted(category_counts.items())),
        "selected_category": "mixed",
        "selected_tasks": len(selected_ids),
        "selected_fraction": len(selected_ids) / len(task_rows),
        "random_control_seed": random_control_seed,
        "random_control_tasks": len(random_control_ids),
        "mean_reward_across_tasks": mean(
            row["reward_mean"] for row in screening_rows
        ),
        "mean_reward_spread": mean(
            row["reward_spread"] for row in screening_rows
        ),
        "artifacts": {
            "screening_jsonl": screening_path.name,
            "screening_sha256": _sha256(screening_path),
            "boundary_tasks_jsonl": selected_path.name,
            "boundary_tasks_sha256": _sha256(selected_path),
            "random_control_tasks_jsonl": random_control_path.name,
            "random_control_tasks_sha256": _sha256(random_control_path),
        },
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen train tasks for GRPO reward diversity"
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--memory", default="512m")
    parser.add_argument("--docker-image", default="python:3.11-slim")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--random-control-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_screening(
        tasks_path=args.tasks,
        predictions_path=args.predictions,
        output_dir=args.output_dir,
        num_samples_per_task=args.num_samples,
        timeout_seconds=args.timeout,
        memory_limit=args.memory,
        docker_image=args.docker_image,
        max_workers=args.max_workers,
        random_control_seed=args.random_control_seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
