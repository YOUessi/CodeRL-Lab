from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from coderl_lab.evaluation import load_tasks
from coderl_lab.execution import PythonExecutor
from coderl_lab.generation import extract_python_code


CONDITIONS = ("low_stabilize", "low_destabilize", "high_stabilize")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_hidden_correct(path: Path) -> dict[str, bool]:
    rows = load_jsonl(path)
    out: dict[str, bool] = {}
    for row in rows:
        task_id = str(row["task_id"])
        if task_id in out:
            raise ValueError(f"duplicate correctness task: {task_id}")
        out[task_id] = bool(row["hidden_all_pass"])
    return out


def load_reference_predictions(path: Path) -> dict[str, str]:
    rows = load_jsonl(path)
    out: dict[str, str] = {}
    for row in rows:
        task_id = str(row["task_id"])
        if task_id in out:
            raise ValueError(f"duplicate reference task: {task_id}")
        out[task_id] = str(row.get("raw_completion", row.get("completion", "")))
    return out


def bootstrap_task_cluster(
    rows: list[dict[str, Any]],
    *,
    value_fn: Callable[[dict[str, Any]], float],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_id"])].append(float(value_fn(row)))
    task_ids = sorted(grouped)
    if not task_ids:
        raise ValueError("no rows for bootstrap")

    task_values = {
        task_id: mean(grouped[task_id])
        for task_id in task_ids
    }
    observed = mean(task_values.values())

    rng = random.Random(seed)
    draws: list[float] = []
    n = len(task_ids)
    for _ in range(iterations):
        draws.append(
            mean(task_values[task_ids[rng.randrange(n)]] for _ in range(n))
        )
    draws.sort()
    lo = draws[int(0.025 * iterations)]
    hi = draws[min(iterations - 1, int(0.975 * iterations))]
    positive = sum(x > 0 for x in draws) / iterations

    return {
        "tasks": n,
        "observed_delta": observed,
        "ci95_low": lo,
        "ci95_high": hi,
        "probability_positive": positive,
        "iterations": iterations,
        "seed": seed,
    }


def transition_counts(
    rows: list[dict[str, Any]],
    condition: str,
) -> dict[str, Any]:
    counts = {
        "wrong_to_correct": 0,
        "correct_to_wrong": 0,
        "correct_to_correct": 0,
        "wrong_to_wrong": 0,
    }
    for row in rows:
        baseline = bool(row["baseline_correct"])
        candidate = bool(row[f"{condition}_correct"])
        if not baseline and candidate:
            counts["wrong_to_correct"] += 1
        elif baseline and not candidate:
            counts["correct_to_wrong"] += 1
        elif baseline and candidate:
            counts["correct_to_correct"] += 1
        else:
            counts["wrong_to_wrong"] += 1
    counts["net_correct_delta"] = (
        counts["wrong_to_correct"] - counts["correct_to_wrong"]
    )
    counts["pairs"] = len(rows)
    return counts


def summarize(
    rows: list[dict[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("no eligible rows")

    result: dict[str, Any] = {
        "eligible_pairs": len(rows),
        "eligible_tasks": len({str(row["task_id"]) for row in rows}),
        "baseline_correct_fraction": mean(
            1.0 if row["baseline_correct"] else 0.0 for row in rows
        ),
        "reference_correct_fraction": mean(
            1.0 if row["reference_correct"] else 0.0 for row in rows
        ),
        "conditions": {},
        "contrasts": {},
    }

    for condition in CONDITIONS:
        correct_fraction = mean(
            1.0 if row[f"{condition}_correct"] else 0.0 for row in rows
        )
        applied_rows = [
            row for row in rows if bool(row[f"{condition}_applied"])
        ]
        result["conditions"][condition] = {
            "correct_fraction": correct_fraction,
            "delta_vs_baseline": correct_fraction
            - result["baseline_correct_fraction"],
            "transitions": transition_counts(rows, condition),
            "intervention_applied_pairs": len(applied_rows),
            "applied_correct_fraction": (
                mean(
                    1.0 if row[f"{condition}_correct"] else 0.0
                    for row in applied_rows
                )
                if applied_rows
                else None
            ),
        }
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

    result["contrasts"]["low_stabilize_minus_high_stabilize"] = (
        bootstrap_task_cluster(
            rows,
            value_fn=lambda row: (
                float(bool(row["low_stabilize_correct"]))
                - float(bool(row["high_stabilize_correct"]))
            ),
            iterations=iterations,
            seed=seed,
        )
    )

    strata: dict[str, Any] = {}
    for name, ref_value in (
        ("reference_correct", True),
        ("reference_wrong", False),
    ):
        subset = [
            row for row in rows
            if bool(row["reference_correct"]) is ref_value
        ]
        strata[name] = {
            "pairs": len(subset),
            "tasks": len({str(row["task_id"]) for row in subset}),
            "baseline_correct_fraction": (
                mean(
                    1.0 if row["baseline_correct"] else 0.0
                    for row in subset
                )
                if subset
                else None
            ),
            "low_stabilize_correct_fraction": (
                mean(
                    1.0 if row["low_stabilize_correct"] else 0.0
                    for row in subset
                )
                if subset
                else None
            ),
            "low_destabilize_correct_fraction": (
                mean(
                    1.0 if row["low_destabilize_correct"] else 0.0
                    for row in subset
                )
                if subset
                else None
            ),
        }
        if subset:
            strata[name]["low_stabilize_minus_baseline"] = (
                bootstrap_task_cluster(
                    subset,
                    value_fn=lambda row: (
                        float(bool(row["low_stabilize_correct"]))
                        - float(bool(row["baseline_correct"]))
                    ),
                    iterations=iterations,
                    seed=seed,
                )
            )
            strata[name]["low_destabilize_minus_baseline"] = (
                bootstrap_task_cluster(
                    subset,
                    value_fn=lambda row: (
                        float(bool(row["low_destabilize_correct"]))
                        - float(bool(row["baseline_correct"]))
                    ),
                    iterations=iterations,
                    seed=seed,
                )
            )
    result["reference_correctness_strata"] = strata

    trajectory_rescues = [
        row for row in rows
        if not bool(row["baseline_exact_reference"])
        and bool(row["low_stabilize_exact_reference"])
    ]
    trajectory_rescue_summary = {
        "pairs": len(trajectory_rescues),
        "reference_correct": sum(
            bool(row["reference_correct"]) for row in trajectory_rescues
        ),
        "reference_wrong": sum(
            not bool(row["reference_correct"]) for row in trajectory_rescues
        ),
        "correctness_rescue": sum(
            (not bool(row["baseline_correct"]))
            and bool(row["low_stabilize_correct"])
            for row in trajectory_rescues
        ),
        "correctness_harm": sum(
            bool(row["baseline_correct"])
            and (not bool(row["low_stabilize_correct"]))
            for row in trajectory_rescues
        ),
    }
    result["trajectory_rescue"] = trajectory_rescue_summary

    induced = [
        row for row in rows
        if bool(row["baseline_exact_reference"])
        and not bool(row["low_destabilize_exact_reference"])
    ]
    result["low_destabilize_induced_divergence"] = {
        "pairs": len(induced),
        "wrong_to_correct": sum(
            (not bool(row["baseline_correct"]))
            and bool(row["low_destabilize_correct"])
            for row in induced
        ),
        "correct_to_wrong": sum(
            bool(row["baseline_correct"])
            and (not bool(row["low_destabilize_correct"]))
            for row in induced
        ),
    }

    return result


def evaluate_condition_codes(
    *,
    rows: list[dict[str, Any]],
    tasks: dict[str, Any],
    executor: PythonExecutor,
    workers: int,
) -> dict[tuple[str, str], bool]:
    unique: dict[tuple[str, str], tuple[str, str]] = {}
    for row in rows:
        task_id = str(row["task_id"])
        task = tasks[task_id]
        for condition in CONDITIONS:
            raw = str(row[f"{condition}_raw_completion"])
            code = extract_python_code(raw, task.entry_point)
            key = (task_id, code)
            unique[key] = (task_id, code)

    def run_one(item: tuple[tuple[str, str], tuple[str, str]]):
        key, (task_id, code) = item
        task = tasks[task_id]
        report = executor.run(
            code,
            entry_point=task.entry_point,
            cases=task.hidden_tests,
            setup_code=task.setup_code,
        )
        return key, bool(report.syntax_ok and report.pass_rate == 1.0)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        evaluated = list(pool.map(run_one, unique.items()))
    return dict(evaluated)


def build_rows(
    *,
    arm_paths: list[Path],
    tasks_path: Path,
    reference_predictions_path: Path,
    reference_details_path: Path,
    greedy_root: Path,
    executor: PythonExecutor,
    workers: int,
) -> list[dict[str, Any]]:
    tasks = load_tasks(tasks_path)
    reference_raw = load_reference_predictions(reference_predictions_path)
    reference_correct = load_hidden_correct(reference_details_path)

    rows: list[dict[str, Any]] = []
    for arm_path in arm_paths:
        arm = json.loads(arm_path.read_text(encoding="utf-8"))
        arm_name = arm_path.stem
        baseline_details = load_hidden_correct(
            greedy_root / arm_name / "evaluation" / "details.jsonl"
        )

        for task_id, item in arm["per_task"].items():
            if not bool(item.get("eligible")):
                continue
            row: dict[str, Any] = {
                "task_id": str(task_id),
                "arm": arm_name,
                "reference_correct": bool(reference_correct[str(task_id)]),
                "baseline_correct": bool(baseline_details[str(task_id)]),
                "reference_raw_completion": reference_raw[str(task_id)],
                "baseline_raw_completion": str(item["baseline"]["raw_completion"]),
                "baseline_exact_reference": bool(
                    item["baseline"]["exact_match_reference"]
                ),
                "low_position": item.get("low_position"),
                "low_margin": item.get("low_margin"),
                "high_control_position": item.get("high_control_position"),
                "high_control_margin": item.get("high_control_margin"),
            }
            for condition in CONDITIONS:
                cond = item[condition]
                row[f"{condition}_raw_completion"] = str(cond["raw_completion"])
                row[f"{condition}_exact_reference"] = bool(
                    cond["exact_match_reference"]
                )
                row[f"{condition}_applied"] = bool(
                    cond.get("intervention_applied", False)
                )
            rows.append(row)

    evaluated = evaluate_condition_codes(
        rows=rows,
        tasks=tasks,
        executor=executor,
        workers=workers,
    )

    for row in rows:
        task_id = str(row["task_id"])
        task = tasks[task_id]
        for condition in CONDITIONS:
            raw = str(row[f"{condition}_raw_completion"])
            if raw == row["baseline_raw_completion"]:
                correct = bool(row["baseline_correct"])
            elif raw == row["reference_raw_completion"]:
                correct = bool(row["reference_correct"])
            else:
                code = extract_python_code(raw, task.entry_point)
                correct = evaluated[(task_id, code)]
            row[f"{condition}_correct"] = bool(correct)

    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate correctness causality of local bottleneck interventions"
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--reference-details", type=Path, required=True)
    parser.add_argument("--greedy-root", type=Path, required=True)
    parser.add_argument("--arm", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--memory", default="512m")
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    executor = PythonExecutor(
        mode="docker",
        timeout_seconds=args.timeout,
        memory_limit=args.memory,
    )
    rows = build_rows(
        arm_paths=args.arm,
        tasks_path=args.tasks,
        reference_predictions_path=args.reference_predictions,
        reference_details_path=args.reference_details,
        greedy_root=args.greedy_root,
        executor=executor,
        workers=args.workers,
    )
    summary = summarize(
        rows,
        iterations=args.iterations,
        seed=args.seed,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "details.jsonl", rows)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
