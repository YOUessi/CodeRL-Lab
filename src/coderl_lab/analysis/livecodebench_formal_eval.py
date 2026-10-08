from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from coderl_lab.analysis.bottleneck_correctness_causality import bootstrap_task_cluster
from coderl_lab.analysis.livecodebench_official_eval import (
    OfficialLiveCodeBenchExecutor,
    decode_test_cases,
    evaluate_code,
    load_public_tasks,
)
from coderl_lab.datasets.livecodebench import sha256_file

CONDITIONS = (
    "baseline",
    "gated_low_destabilize",
    "gated_high_destabilize",
    "always_low_destabilize",
)
SOURCE_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
OFFICIAL_COMMIT = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_frozen_input(
    runner: dict[str, Any],
    public_rows: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
    *,
    expected_tasks: int,
    lcb_repo: Path,
    runner_path: Path,
) -> dict[str, Any]:
    if expected_tasks not in (12, 175):
        raise ValueError("EXP-004L requires exactly 12 smoke or 175 formal tasks")
    if runner["experiment"] != "EXP-004L":
        raise ValueError("wrong experiment")
    if runner.get("private_tests_accessed") is not False:
        raise ValueError("private-test isolation attestation missing")
    if runner.get("official_livecodebench_commit") != OFFICIAL_COMMIT:
        raise ValueError("official evaluator commit mismatch")
    if runner.get("tasks") != expected_tasks:
        raise ValueError(f"expected {expected_tasks} frozen tasks, found {runner.get('tasks')}")
    if manifest.get("source_revision") != SOURCE_REVISION:
        raise ValueError("source revision drift")
    if manifest.get("tasks") != 175:
        raise ValueError("full data manifest count must be 175")
    if manifest.get("private_tests_exported") is not False:
        raise ValueError("manifest contains private-test export")
    if manifest.get("private_tests_decoded") is not False:
        raise ValueError("private tests were decoded during data preparation")
    if len(public_rows) != 175:
        raise ValueError("public source must contain 175 tasks")

    git = subprocess.run(
        ["git", "-C", str(lcb_repo), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=True,
    )
    if git.stdout.strip() != OFFICIAL_COMMIT:
        raise ValueError("official checker checkout changed")

    for key, wanted in (
        ("primary_window", 128),
        ("low_threshold", 0.05),
        ("high_threshold", 0.20),
        ("bias", 0.25),
        ("max_new_tokens", 512),
    ):
        if runner.get(key) != wanted:
            raise ValueError(f"pre-registered decoding rule drift: {key}")

    tasks = runner["per_task"]
    if len(tasks) != expected_tasks or set(tasks) - set(public_rows):
        raise ValueError("missing/extra or duplicate runner tasks")
    if expected_tasks == 175 and set(tasks) != set(public_rows):
        raise ValueError("formal runner did not cover the full v6 distribution")

    for qid, task in tasks.items():
        if task.get("question_id") != qid:
            raise ValueError(f"runner question ID mismatch: {qid}")
        public_ok = bool(task["public_all_pass"])
        eligible = bool(task["eligible"])
        gated = bool(task["gate_triggered"])
        if gated != ((not public_ok) and eligible):
            raise ValueError(f"gate is inconsistent with public-only rule: {qid}")
        baseline = str(task["baseline"]["raw_completion"])
        if not gated:
            for condition in ("gated_low_destabilize", "gated_high_destabilize"):
                if str(task[condition]["raw_completion"]) != baseline:
                    raise ValueError(f"gate erroneously rewrote {qid}: {condition}")
        if not eligible and str(task["always_low_destabilize"]["raw_completion"]) != baseline:
            raise ValueError(f"non-eligible task was modified: {qid}")
        if str(task["baseline"]["code"]) != baseline.strip():
            raise ValueError(f"baseline postprocessing drift: {qid}")

    return {
        "experiment": "EXP-004L",
        "phase": "frozen-before-private-tests",
        "expected_tasks": expected_tasks,
        "runner_sha256": sha256_file(runner_path),
        "dataset_source_revision": SOURCE_REVISION,
        "dataset_source_sha256": manifest["source_sha256"],
        "public_view_sha256": manifest["public_view_sha256"],
        "official_evaluator_commit": OFFICIAL_COMMIT,
        "private_tests_accessed": False,
    }


def transitions(rows: list[dict[str, Any]], condition: str) -> dict[str, int]:
    pairs = Counter()
    for row in rows:
        b = bool(row["baseline_correct"])
        c = bool(row[f"{condition}_correct"])
        if not b and c:
            pairs["wrong_to_correct"] += 1
        elif b and not c:
            pairs["correct_to_wrong"] += 1
        elif b and c:
            pairs["correct_to_correct"] += 1
        else:
            pairs["wrong_to_wrong"] += 1
    return {name: pairs[name] for name in (
        "wrong_to_correct", "correct_to_wrong",
        "correct_to_correct", "wrong_to_wrong",
    )}


def summarize(
    rows: list[dict[str, Any]], *, iterations: int, seed: int
) -> dict[str, Any]:
    if not rows:
        raise ValueError("empty evaluation")
    summary: dict[str, Any] = {
        "tasks": len(rows),
        "eligible_tasks": sum(bool(row["eligible"]) for row in rows),
        "gate_triggered_tasks": sum(bool(row["gate_triggered"]) for row in rows),
        "public_pass_tasks": sum(bool(row["public_all_pass"]) for row in rows),
        "conditions": {},
        "contrasts": {},
        "strata": {},
    }
    for condition in CONDITIONS:
        correct = sum(bool(row[f"{condition}_correct"]) for row in rows)
        summary["conditions"][condition] = {
            "correct_tasks": correct,
            "correct_fraction": correct / len(rows),
            "delta_vs_baseline": None,
            "transitions_vs_baseline": (
                None if condition == "baseline" else transitions(rows, condition)
            ),
        }

    baseline = summary["conditions"]["baseline"]["correct_fraction"]
    for condition in CONDITIONS[1:]:
        summary["conditions"][condition]["delta_vs_baseline"] = (
            summary["conditions"][condition]["correct_fraction"] - baseline
        )
        summary["contrasts"][f"{condition}_minus_baseline"] = bootstrap_task_cluster(
            rows,
            value_fn=lambda row, c=condition: (
                int(row[f"{c}_correct"]) - int(row["baseline_correct"])
            ),
            iterations=iterations, seed=seed,
        )
    for left, right in (
        ("gated_low_destabilize", "gated_high_destabilize"),
        ("gated_low_destabilize", "always_low_destabilize"),
    ):
        summary["contrasts"][f"{left}_minus_{right}"] = bootstrap_task_cluster(
            rows,
            value_fn=lambda row, a=left, b=right: (
                int(row[f"{a}_correct"]) - int(row[f"{b}_correct"])
            ),
            iterations=iterations, seed=seed,
        )

    gated = [row for row in rows if bool(row["gate_triggered"])]
    if gated:
        summary["gate_triggered_subset"] = {
            "tasks": len(gated),
            "baseline_correct": sum(bool(x["baseline_correct"]) for x in gated),
            "gated_low_correct": sum(
                bool(x["gated_low_destabilize_correct"]) for x in gated
            ),
            "low_transitions": transitions(gated, "gated_low_destabilize"),
            "low_minus_baseline": bootstrap_task_cluster(
                gated,
                value_fn=lambda x: (
                    int(x["gated_low_destabilize_correct"]) - int(x["baseline_correct"])
                ),
                iterations=iterations, seed=seed,
            ),
        }

    public_pass = [row for row in rows if bool(row["public_all_pass"])]
    summary["public_pass_protection"] = {
        "tasks": len(public_pass),
        "gated_low_changed": sum(bool(r["gated_low_destabilize_changed"]) for r in public_pass),
        "gated_high_changed": sum(bool(r["gated_high_destabilize_changed"]) for r in public_pass),
        "always_low_changed": sum(bool(r["always_low_destabilize_changed"]) for r in public_pass),
    }

    wrong = [row for row in rows if not bool(row["baseline_correct"])]
    selected_wrong = sum(not bool(r["baseline_correct"]) for r in gated)
    public_fail = [r for r in rows if not bool(r["public_all_pass"])]
    summary["posthoc_gate_audit"] = {
        "baseline_wrong_tasks": len(wrong),
        "gate_selected_tasks": len(gated),
        "gate_selected_hidden_wrong": selected_wrong,
        "gate_precision": selected_wrong / len(gated) if gated else None,
        "gate_recall": selected_wrong / len(wrong) if wrong else None,
        "public_fail_precision": (
            sum(not bool(r["baseline_correct"]) for r in public_fail) / len(public_fail)
            if public_fail else None
        ),
    }

    for field in ("platform", "difficulty"):
        for value in sorted({str(row[field]) for row in rows}):
            subset = [row for row in rows if str(row[field]) == value]
            key = f"{field}:{value}"
            summary["strata"][key] = {
                "tasks": len(subset),
                "baseline_correct": sum(bool(x["baseline_correct"]) for x in subset),
                "gated_low_correct": sum(
                    bool(x["gated_low_destabilize_correct"]) for x in subset
                ),
            }

    ci = summary["contrasts"]["gated_low_destabilize_minus_baseline"]
    summary["primary_success"] = ci["ci95_low"] > 0
    summary["claim_limit"] = (
        "formal 175-task external-distribution replication"
        if len(rows) == 175 else "engineering smoke only, not efficacy evidence"
    )
    return summary


def run_final(
    *,
    runner_path: Path,
    public_path: Path,
    manifest_path: Path,
    source_path: Path,
    lcb_repo: Path,
    output_dir: Path,
    expected_tasks: int,
    iterations: int,
    seed: int,
    timeout: int,
    memory: str,
) -> dict[str, Any]:
    runner = _json(runner_path)
    public = load_public_tasks(public_path)
    manifest = _json(manifest_path)
    if sha256_file(public_path) != manifest["public_view_sha256"]:
        raise ValueError("public view SHA differs from frozen manifest")
    frozen = validate_frozen_input(
        runner, public, manifest, expected_tasks=expected_tasks,
        lcb_repo=lcb_repo, runner_path=runner_path,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = output_dir / "frozen_inputs.json"
    if freeze_path.exists():
        if _json(freeze_path) != frozen:
            raise ValueError("frozen runner or dataset changed since private evaluation")
    else:
        freeze_path.write_text(
            json.dumps(frozen, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    # The first access to private test payloads occurs BELOW the freeze barrier.
    if sha256_file(source_path) != manifest["source_sha256"]:
        raise ValueError("upstream source checksum changed after freeze")

    checkpoint = output_dir / "private_evaluation_checkpoint.jsonl"
    cache: dict[tuple[str, str], bool] = {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            key = (str(item["question_id"]), str(item["code_sha256"]))
            if key in cache and cache[key] != bool(item["passed"]):
                raise ValueError("conflicting checkpoint result")
            cache[key] = bool(item["passed"])

    evaluator = OfficialLiveCodeBenchExecutor(
        livecodebench_repo=lcb_repo, timeout_seconds=timeout, memory_limit=memory,
    )

    required = set(runner["per_task"])
    found: set[str] = set()
    for line in source_path.open(encoding="utf-8"):
        if not line.strip():
            continue
        source_row = json.loads(line)
        qid = str(source_row["question_id"])
        if qid not in required:
            continue
        if qid in found:
            raise ValueError(f"duplicate source question {qid}")
        found.add(qid)
        private_tests = decode_test_cases(source_row["private_test_cases"])
        if not private_tests:
            raise ValueError(f"no private cases: {qid}")
        task = runner["per_task"][qid]
        codes = {
            condition: str(task[condition]["code"])
            for condition in CONDITIONS
        }
        for condition in CONDITIONS:
            code = codes[condition]
            key = (qid, _sha_text(code))
            if key in cache:
                continue
            outcome = evaluate_code(
                executor=evaluator, task=public[qid],
                code=code, private_tests=private_tests,
            )
            if outcome.results and str(outcome.results[0]) in (
                "runner-error", "invalid-runner-output", "global-timeout",
            ):
                raise RuntimeError(
                    f"evaluator infrastructure failure at {qid}/{condition}; "
                    "not counting this as model error"
                )
            record = {
                "question_id": qid,
                "code_sha256": key[1],
                "passed": outcome.passed,
                "number_of_results": len(outcome.results),
            }
            # Never persist the official error metadata: it may contain private I/O.
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            cache[key] = outcome.passed
        print(f"EXP-004L final: {len(found)}/{len(required)} tasks", flush=True)
    if found != required:
        raise ValueError(f"source is missing {sorted(required - found)}")

    detail_rows: list[dict[str, Any]] = []
    for qid in sorted(required):
        task = runner["per_task"][qid]
        baseline_code = str(task["baseline"]["code"])
        row: dict[str, Any] = {
            "task_id": qid,
            "platform": str(task["platform"]),
            "difficulty": str(task["difficulty"]),
            "eligible": bool(task["eligible"]),
            "gate_triggered": bool(task["gate_triggered"]),
            "public_all_pass": bool(task["public_all_pass"]),
        }
        for condition in CONDITIONS:
            code = str(task[condition]["code"])
            row[f"{condition}_correct"] = cache[(qid, _sha_text(code))]
            row[f"{condition}_changed"] = (
                str(task[condition]["raw_completion"])
                != str(task["baseline"]["raw_completion"])
            )
        detail_rows.append(row)

    result = summarize(detail_rows, iterations=iterations, seed=seed)
    result["frozen_inputs"] = frozen
    result["private_tests_accessed_only_after_freeze"] = True

    with (output_dir / "details.jsonl").open("w", encoding="utf-8") as handle:
        for row in detail_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "tasks": result["tasks"],
        "conditions": result["conditions"],
        "primary": result["contrasts"]["gated_low_destabilize_minus_baseline"],
        "primary_success": result["primary_success"],
    }, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--public-tasks", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path)
    parser.add_argument("--lcb-repo", type=Path, default=Path(".external/livecodebench"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, choices=(12, 175), required=True)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--memory", default="1g")
    args = parser.parse_args()
    if args.source_jsonl is None:
        from huggingface_hub import hf_hub_download
        # Downloading is allowed here, but the first decode of PRIVATE CASES
        # remains below run_final's frozen-input audit.
        source = Path(hf_hub_download(
            repo_id="livecodebench/code_generation_lite",
            filename="test6.jsonl", repo_type="dataset",
            revision=SOURCE_REVISION,
        ))
    else:
        source = args.source_jsonl

    run_final(
        runner_path=args.runner, public_path=args.public_tasks,
        manifest_path=args.manifest, source_path=source,
        lcb_repo=args.lcb_repo, output_dir=args.output_dir,
        expected_tasks=args.expected_tasks, iterations=args.iterations,
        seed=args.seed, timeout=args.timeout, memory=args.memory,
    )


if __name__ == "__main__":
    main()
