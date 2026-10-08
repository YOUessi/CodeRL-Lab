"""Retrospective EXP-004M trace effectiveness audit (CPU only).

Reads the already frozen EXP-004L runner and private correctness summary;
emits *aggregate metadata only*, never private test inputs, outputs, or code.
It does not generate new candidates or select thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

FIELDS = ("baseline", "gated_low_destabilize", "gated_high_destabilize", "always_low_destabilize")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _location_bucket(position: int | None) -> str:
    if position is None:
        return "none"
    if position < 0 or position >= 128:
        raise ValueError("low-margin target outside registered 128-token window")
    if position < 32:
        return "00-31"
    if position < 64:
        return "32-63"
    return "64-127"


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    n = len(rows)
    gate_rows = [r for r in rows if r["gate"]]
    changed = [r for r in gate_rows if r["low_changed"]]
    eligible = [r for r in rows if r["eligible"]]
    return {
        "n": n,
        "eligible": len(eligible),
        "gate_triggered": len(gate_rows),
        "gate_fraction": len(gate_rows) / n,
        "low_applied_on_gated": sum(r["low_applied"] for r in gate_rows),
        "low_changed_on_gated": len(changed),
        "low_changed_fraction_of_gated": len(changed) / len(gate_rows) if gate_rows else None,
        "high_changed_on_gated": sum(r["high_changed"] for r in gate_rows),
        "rescue_on_gated": sum(r["rescue"] for r in gate_rows),
        "harm_on_gated": sum(r["harm"] for r in gate_rows),
        "rescues_among_changed": sum(r["rescue"] for r in changed),
        "rescue_fraction_of_changed": (
            sum(r["rescue"] for r in changed) / len(changed) if changed else None
        ),
        "baseline_correct": sum(r["base_correct"] for r in rows),
        "low_correct": sum(r["low_correct"] for r in rows),
    }


def summarize_trace(
    runner: dict[str, Any],
    detail_rows: list[dict[str, Any]],
    frozen_summary: dict[str, Any],
    *,
    runner_sha256: str,
) -> dict[str, Any]:
    provenance = frozen_summary.get("frozen_inputs", {})
    if runner.get("experiment") != "EXP-004L":
        raise ValueError("not an EXP-004L runner")
    if runner.get("private_tests_accessed") is not False:
        raise ValueError("runner isolation attestation missing")
    if not frozen_summary.get("private_tests_accessed_only_after_freeze"):
        raise ValueError("final evaluation isolation attestation missing")
    if provenance.get("runner_sha256") != runner_sha256:
        raise ValueError("frozen runner hash differs from final evaluation")
    expected = frozen_summary.get("tasks")
    if not isinstance(expected, int) or expected != runner.get("tasks"):
        raise ValueError("runner/final task count mismatch")
    records = runner.get("per_task", {})
    if len(records) != expected:
        raise ValueError("incorrect frozen runner task count")
    if len(detail_rows) != expected:
        raise ValueError("incorrect final result task count")

    correct: dict[str, dict[str, Any]] = {}
    for row in detail_rows:
        tid = str(row["task_id"])
        if tid in correct:
            raise ValueError(f"duplicate final result task {tid}")
        correct[tid] = row
    if set(correct) != set(records):
        raise ValueError("runner and final task IDs differ")

    rows: list[dict[str, Any]] = []
    for tid in sorted(records):
        item, outcome = records[tid], correct[tid]
        gate, eligible = bool(item["gate_triggered"]), bool(item["eligible"])
        public_pass = bool(item["public_all_pass"])
        if gate != (not public_pass and eligible):
            raise ValueError(f"gate invariant broken for {tid}")
        if gate != bool(outcome["gate_triggered"]) or eligible != bool(outcome["eligible"]):
            raise ValueError(f"gate/eligibility result mismatch for {tid}")
        raw = str(item["baseline"]["raw_completion"])
        low = str(item["gated_low_destabilize"]["raw_completion"])
        high = str(item["gated_high_destabilize"]["raw_completion"])
        if not gate and (raw != low or raw != high):
            raise ValueError(f"non-triggered task changed for {tid}")
        if not gate and bool(item["gated_low_destabilize"].get("intervention_applied")):
            raise ValueError(f"non-triggered token intervention for {tid}")
        if str(item["platform"]) != str(outcome["platform"]) or str(item["difficulty"]) != str(outcome["difficulty"]):
            raise ValueError(f"task stratum mismatch for {tid}")

        base_correct = bool(outcome["baseline_correct"])
        low_correct = bool(outcome["gated_low_destabilize_correct"])
        rows.append({
            "platform": str(item["platform"]),
            "difficulty": str(item["difficulty"]),
            "gate": gate,
            "eligible": eligible,
            "target_bucket": _location_bucket(item["low_position"]),
            "low_applied": bool(item["gated_low_destabilize"].get("intervention_applied", False)),
            "low_changed": low != raw,
            "high_changed": high != raw,
            "base_correct": base_correct,
            "low_correct": low_correct,
            "rescue": (not base_correct) and low_correct,
            "harm": base_correct and (not low_correct),
        })

    total = _aggregate(rows)
    official = frozen_summary["conditions"]
    if total["baseline_correct"] != official["baseline"]["correct_tasks"]:
        raise ValueError("baseline correctness does not reconcile")
    if total["low_correct"] != official["gated_low_destabilize"]["correct_tasks"]:
        raise ValueError("gated low correctness does not reconcile")
    if total["gate_triggered"] != frozen_summary["gate_triggered_tasks"]:
        raise ValueError("gate triggered count does not reconcile")
    transition = official["gated_low_destabilize"]["transitions_vs_baseline"]
    if total["rescue_on_gated"] != transition["wrong_to_correct"] or total["harm_on_gated"] != transition["correct_to_wrong"]:
        raise ValueError("frozen primary transitions do not reconcile")

    groupings: dict[str, dict[str, Any]] = {}
    for name in ("platform", "difficulty", "target_bucket"):
        keys = sorted(set(row[name] for row in rows))
        groupings[name] = {
            key: _aggregate([r for r in rows if r[name] == key]) for key in keys
        }
    groupings["gate_state"] = {
        "triggered": _aggregate([r for r in rows if r["gate"]]),
        "not_triggered": _aggregate([r for r in rows if not r["gate"]]),
    }
    return {
        "experiment": "EXP-004M",
        "audit": "posthoc_trace_effectiveness",
        "source": "frozen_EXP-004L",
        "runner_sha256": runner_sha256,
        "task_count": expected,
        "overall": total,
        "groups": groupings,
        "frozen_intervention_parameters_unchanged": True,
        "hidden_test_cases_or_generated_code_exported": False,
        "causal_interpretation": (
            "Exploratory trace-level accounting only; a high changed fraction "
            "with low rescue would be consistent with trajectory-quality / model "
            "capability bottlenecks, not causal proof of a floor effect."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--official-details", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = summarize_trace(
        json.loads(args.runner.read_text(encoding="utf-8")),
        _load_jsonl(args.official_details),
        json.loads(args.summary.read_text(encoding="utf-8")),
        runner_sha256=sha256_file(args.runner),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # Print *aggregates only* so GitHub can store evidence without raw private
    # tests, individual generated candidate code, or task-level outputs.
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
