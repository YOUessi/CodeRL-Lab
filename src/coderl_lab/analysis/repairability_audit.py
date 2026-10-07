from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from coderl_lab.data.runtime_repair_preferences import (
    apply_import_repair,
    import_lines_for,
)
from coderl_lab.execution import PythonExecutor
from coderl_lab.reward import analyze_dependency_completeness
from coderl_lab.schema import TestCase


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit(
    *,
    tasks_path: Path,
    predictions_path: Path,
    details_path: Path,
    workers: int,
    timeout_seconds: float,
    memory_limit: str,
) -> dict[str, Any]:
    tasks = {str(x["task_id"]): x for x in load_jsonl(tasks_path)}
    predictions = {
        (str(x["task_id"]), int(x["sample_id"])): x
        for x in load_jsonl(predictions_path)
    }
    details = {
        (str(x["task_id"]), int(x["sample_id"])): x
        for x in load_jsonl(details_path)
    }

    executor = PythonExecutor(
        mode="docker",
        timeout_seconds=timeout_seconds,
        memory_limit=memory_limit,
    )

    candidates: list[dict[str, Any]] = []
    unresolved_counter: Counter[str] = Counter()
    skipped: Counter[str] = Counter()

    for key, pred in predictions.items():
        task = tasks[key[0]]
        detail = details[key]
        public_rate = float(detail["public"]["pass_rate"])
        if public_rate >= 1.0:
            skipped["already_public_all_pass"] += 1
            continue

        code = str(pred["completion"])
        dep = analyze_dependency_completeness(
            code,
            entry_point=str(task["entry_point"]),
            setup_code=str(task.get("setup_code", "")),
        )
        if not dep.unresolved_names:
            skipped["no_unresolved_names"] += 1
            continue

        imports = import_lines_for(dep.unresolved_names)
        if imports is None:
            skipped["unpatchable_name"] += 1
            continue

        for name in dep.unresolved_names:
            unresolved_counter[str(name)] += 1

        candidates.append(
            {
                "task_id": key[0],
                "sample_id": key[1],
                "entry_point": str(task["entry_point"]),
                "setup_code": str(task.get("setup_code", "")),
                "public_tests": list(task["public_tests"]),
                "original_public_pass_rate": public_rate,
                "unresolved_names": list(dep.unresolved_names),
                "import_lines": imports,
                "patched_code": apply_import_repair(code, imports),
            }
        )

    def score(row: dict[str, Any]) -> dict[str, Any]:
        report = executor.run(
            str(row["patched_code"]),
            entry_point=str(row["entry_point"]),
            cases=tuple(TestCase.from_dict(x) for x in row["public_tests"]),
            setup_code=str(row["setup_code"]),
        )
        out = dict(row)
        out["patched_syntax_ok"] = bool(report.syntax_ok)
        out["patched_public_pass_rate"] = float(report.pass_rate)
        out["verified_repair"] = bool(
            report.syntax_ok and report.pass_rate == 1.0
        )
        return out

    if workers == 1:
        scored = [score(row) for row in candidates]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            scored = list(pool.map(score, candidates))

    verified = [row for row in scored if row["verified_repair"]]
    verified_tasks = {str(row["task_id"]) for row in verified}
    import_counter: Counter[str] = Counter()
    for row in verified:
        for line in row["import_lines"]:
            import_counter[str(line)] += 1

    total = len(predictions)
    public_failures = total - skipped["already_public_all_pass"]
    return {
        "num_predictions": total,
        "public_failure_candidates": public_failures,
        "standardlib_patch_candidates": len(candidates),
        "verified_minimal_import_repairs": len(verified),
        "verified_distinct_tasks": len(verified_tasks),
        "repairable_fraction_of_predictions": (
            len(verified) / total if total else 0.0
        ),
        "repairable_fraction_of_public_failures": (
            len(verified) / public_failures if public_failures else 0.0
        ),
        "top_unresolved_names_in_patch_candidates": unresolved_counter.most_common(20),
        "verified_import_counts": dict(import_counter.most_common()),
        "skipped": dict(skipped),
        "hidden_tests_used": False,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit minimal-import repairable failures using public tests only"
    )
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--details", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--memory", default="512m")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(
        tasks_path=args.tasks,
        predictions_path=args.predictions,
        details_path=args.details,
        workers=args.workers,
        timeout_seconds=args.timeout,
        memory_limit=args.memory,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
