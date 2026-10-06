from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from coderl_lab.metrics import mean_pass_at_k


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_overlap(
    coderl_tasks: list[dict[str, Any]],
    plus_problems: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    plus_by_source = {
        int(task_id.split("/")[-1]): problem
        for task_id, problem in plus_problems.items()
    }

    overlap: list[dict[str, Any]] = []
    for task in coderl_tasks:
        source_id = int(task["metadata"]["source_task_id"])
        problem = plus_by_source.get(source_id)
        if problem is None:
            continue
        if str(task["entry_point"]) != str(problem["entry_point"]):
            raise ValueError(
                f"entry point mismatch for MBPP {source_id}: "
                f"CodeRL={task['entry_point']}, EvalPlus={problem['entry_point']}"
            )
        overlap.append(
            {
                "source_task_id": source_id,
                "coderl_task_id": str(task["task_id"]),
                "evalplus_task_id": str(problem["task_id"]),
                "entry_point": str(problem["entry_point"]),
                "base_tests": len(problem["base_input"]),
                "plus_tests": len(problem["plus_input"]),
            }
        )
    return overlap


def _evaluate_one(args: tuple[Any, ...]) -> dict[str, Any]:
    (
        task_id,
        sample_id,
        problem,
        solution,
        expected,
        min_time_limit,
        gt_time_limit_factor,
    ) = args

    from evalplus.eval import PASS
    from evalplus.evaluate import check_correctness

    result = check_correctness(
        "mbpp",
        sample_id,
        problem,
        solution,
        expected,
        base_only=False,
        fast_check=True,
        identifier=f"{task_id}:{sample_id}",
        min_time_limit=min_time_limit,
        gt_time_limit_factor=gt_time_limit_factor,
    )
    base_status = result["base"][0]
    plus_status = result["plus"][0]
    return {
        "task_id": task_id,
        "sample_id": sample_id,
        "base_status": base_status,
        "plus_status": plus_status,
        "base_correct": base_status == PASS,
        "plus_correct": base_status == plus_status == PASS,
    }


def summarize(
    rows: list[dict[str, Any]],
    ks: tuple[int, ...],
) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)

    base_counts: list[tuple[int, int]] = []
    plus_counts: list[tuple[int, int]] = []
    per_task: dict[str, Any] = {}

    for task_id, task_rows in sorted(by_task.items()):
        task_rows.sort(key=lambda x: int(x["sample_id"]))
        n = len(task_rows)
        base_c = sum(bool(x["base_correct"]) for x in task_rows)
        plus_c = sum(bool(x["plus_correct"]) for x in task_rows)
        base_counts.append((n, base_c))
        plus_counts.append((n, plus_c))
        per_task[task_id] = {
            "samples": n,
            "base_correct": base_c,
            "plus_correct": plus_c,
            "base_success_rate": base_c / n,
            "plus_success_rate": plus_c / n,
            "robustness_gap": (base_c - plus_c) / n,
        }

    base_pass = {}
    plus_pass = {}
    for k in ks:
        if all(n >= k for n, _ in base_counts):
            base_pass[f"pass@{k}"] = mean_pass_at_k(base_counts, k)
            plus_pass[f"pass@{k}"] = mean_pass_at_k(plus_counts, k)

    return {
        "num_tasks": len(by_task),
        "num_predictions": len(rows),
        "base_pass_at_k": base_pass,
        "plus_pass_at_k": plus_pass,
        "base_solved_tasks": sum(v["base_correct"] > 0 for v in per_task.values()),
        "plus_solved_tasks": sum(v["plus_correct"] > 0 for v in per_task.values()),
        "mean_base_success_rate": (
            sum(v["base_success_rate"] for v in per_task.values()) / len(per_task)
        ),
        "mean_plus_success_rate": (
            sum(v["plus_success_rate"] for v in per_task.values()) / len(per_task)
        ),
        "mean_robustness_gap": (
            sum(v["robustness_gap"] for v in per_task.values()) / len(per_task)
        ),
        "per_task": per_task,
    }


def evaluate_mbppplus_overlap(
    *,
    tasks_path: Path,
    predictions_path: Path,
    output_dir: Path,
    max_tasks: int | None,
    max_samples_per_task: int | None,
    workers: int,
    version: str,
    min_time_limit: float,
    gt_time_limit_factor: float,
    ks: tuple[int, ...],
) -> dict[str, Any]:
    from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    from evalplus.evaluate import get_groundtruth

    plus = get_mbpp_plus(version=version)
    plus_hash = get_mbpp_plus_hash(version=version)
    coderl_tasks = load_jsonl(tasks_path)
    overlap = build_overlap(coderl_tasks, plus)

    if max_tasks is not None:
        overlap = overlap[:max_tasks]
    if not overlap:
        raise ValueError("No compatible MBPP+ overlap tasks found")

    coderl_to_plus = {
        row["coderl_task_id"]: row["evalplus_task_id"] for row in overlap
    }
    selected_plus_ids = set(coderl_to_plus.values())
    subset = {task_id: plus[task_id] for task_id in selected_plus_ids}

    overlap_key = ",".join(sorted(selected_plus_ids))
    cache_hash = hashlib.sha256(
        f"{plus_hash}|{overlap_key}".encode("utf-8")
    ).hexdigest()
    expected = get_groundtruth(
        subset,
        f"coderl-mbppplus-{cache_hash}",
        MBPP_OUTPUT_NOT_NONE_TASKS,
    )

    predictions = load_jsonl(predictions_path)
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)

    for row in predictions:
        coderl_task_id = str(row["task_id"])
        plus_task_id = coderl_to_plus.get(coderl_task_id)
        if plus_task_id is None:
            continue
        if (
            max_samples_per_task is not None
            and counts[plus_task_id] >= max_samples_per_task
        ):
            continue
        selected.append(
            {
                "plus_task_id": plus_task_id,
                "sample_id": int(row["sample_id"]),
                "solution": str(row["completion"]),
            }
        )
        counts[plus_task_id] += 1

    expected_task_ids = set(selected_plus_ids)
    if set(counts) != expected_task_ids:
        missing = sorted(expected_task_ids - set(counts))
        raise ValueError(f"predictions missing overlap tasks: {missing}")

    jobs = [
        (
            row["plus_task_id"],
            row["sample_id"],
            subset[row["plus_task_id"]],
            row["solution"],
            expected[row["plus_task_id"]],
            min_time_limit,
            gt_time_limit_factor,
        )
        for row in selected
    ]

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_evaluate_one, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda x: (x["task_id"], int(x["sample_id"])))
    summary = summarize(results, ks)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "details.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "evalplus_version": __import__("evalplus").__version__,
        "mbpp_plus_version": __import__(
            "evalplus.data.mbpp", fromlist=["MBPP_PLUS_VERSION"]
        ).MBPP_PLUS_VERSION,
        "mbpp_plus_hash": plus_hash,
        "coderl_tasks": len(coderl_tasks),
        "evalplus_tasks": len(plus),
        "overlap_tasks": len(overlap),
        "overlap": overlap,
        "workers": workers,
        "min_time_limit": min_time_limit,
        "gt_time_limit_factor": gt_time_limit_factor,
        "max_samples_per_task": max_samples_per_task,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"summary": summary, "metadata": metadata}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate fixed CodeRL-Lab predictions on MBPP+ overlap"
    )
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-tasks", type=int)
    p.add_argument("--max-samples-per-task", type=int)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--version", default="v0.2.0")
    p.add_argument("--min-time-limit", type=float, default=1.0)
    p.add_argument("--gt-time-limit-factor", type=float, default=4.0)
    p.add_argument("--k", type=int, nargs="+", default=[1, 4, 8, 16])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_mbppplus_overlap(
        tasks_path=args.tasks,
        predictions_path=args.predictions,
        output_dir=args.output,
        max_tasks=args.max_tasks,
        max_samples_per_task=args.max_samples_per_task,
        workers=args.workers,
        version=args.version,
        min_time_limit=args.min_time_limit,
        gt_time_limit_factor=args.gt_time_limit_factor,
        ks=tuple(args.k),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
