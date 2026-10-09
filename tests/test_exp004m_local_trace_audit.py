from __future__ import annotations

import copy

import pytest

from coderl_lab.analysis.exp004m_local_trace_audit import summarize_trace


def fixtures():
    tasks = {
        "a": {
            "platform": "atcoder",
            "difficulty": "easy",
            "eligible": True, "gate_triggered": True, "public_all_pass": False,
            "low_position": 17,
            "baseline": {"raw_completion": "A"},
            "gated_low_destabilize": {"raw_completion": "B", "intervention_applied": True},
            "gated_high_destabilize": {"raw_completion": "A"},
        },
        "b": {
            "platform": "leetcode",
            "difficulty": "hard",
            "eligible": True, "gate_triggered": True, "public_all_pass": False,
            "low_position": 83,
            "baseline": {"raw_completion": "C"},
            "gated_low_destabilize": {"raw_completion": "D", "intervention_applied": True},
            "gated_high_destabilize": {"raw_completion": "C"},
        },
        "c": {
            "platform": "atcoder",
            "difficulty": "easy",
            "eligible": False, "gate_triggered": False, "public_all_pass": True,
            "low_position": None,
            "baseline": {"raw_completion": "E"},
            "gated_low_destabilize": {"raw_completion": "E", "intervention_applied": False},
            "gated_high_destabilize": {"raw_completion": "E"},
        },
    }
    runner = {
        "experiment": "EXP-004L",
        "private_tests_accessed": False,
        "tasks": 3,
        "per_task": tasks,
    }
    details = [
        {"task_id": "a", "platform": "atcoder", "difficulty": "easy",
         "eligible": True, "gate_triggered": True,
         "baseline_correct": False, "gated_low_destabilize_correct": True},
        {"task_id": "b", "platform": "leetcode", "difficulty": "hard",
         "eligible": True, "gate_triggered": True,
         "baseline_correct": False, "gated_low_destabilize_correct": False},
        {"task_id": "c", "platform": "atcoder", "difficulty": "easy",
         "eligible": False, "gate_triggered": False,
         "baseline_correct": True, "gated_low_destabilize_correct": True},
    ]
    summary = {
        "tasks": 3,
        "private_tests_accessed_only_after_freeze": True,
        "frozen_inputs": {"runner_sha256": "fixed-sha"},
        "gate_triggered_tasks": 2,
        "conditions": {
            "baseline": {"correct_tasks": 1},
            "gated_low_destabilize": {
                "correct_tasks": 2,
                "transitions_vs_baseline": {"wrong_to_correct": 1, "correct_to_wrong": 0},
            },
        },
    }
    return runner, details, summary


def test_trace_audit_matches_official_correctness_and_changed_counts() -> None:
    runner, details, summary = fixtures()
    out = summarize_trace(runner, details, summary, runner_sha256="fixed-sha")
    assert out["task_count"] == 3
    assert out["overall"]["low_changed_on_gated"] == 2
    assert out["overall"]["rescue_on_gated"] == 1
    assert out["overall"]["rescue_fraction_of_changed"] == 0.5
    assert out["groups"]["target_bucket"]["00-31"]["rescue_on_gated"] == 1
    assert out["groups"]["target_bucket"]["64-127"]["rescue_on_gated"] == 0
    assert out["groups"]["platform"]["leetcode"]["low_changed_on_gated"] == 1
    assert out["hidden_test_cases_or_generated_code_exported"] is False
    assert "raw_completion" not in str(out)


def test_rejects_wrong_input_hash() -> None:
    runner, details, summary = fixtures()
    with pytest.raises(ValueError, match="hash"):
        summarize_trace(runner, details, summary, runner_sha256="tampered")


def test_rejects_non_gated_behavior_modification() -> None:
    runner, details, summary = fixtures()
    runner["per_task"]["c"]["gated_low_destabilize"]["raw_completion"] = "unexpected"
    with pytest.raises(ValueError, match="non-triggered"):
        summarize_trace(runner, details, summary, runner_sha256="fixed-sha")


def test_rejects_task_missing_and_duplicate_outcome() -> None:
    runner, details, summary = fixtures()
    with pytest.raises(ValueError, match="task count"):
        summarize_trace(runner, details[:2], summary, runner_sha256="fixed-sha")
    dup = details[:2] + [copy.deepcopy(details[1])]
    with pytest.raises(ValueError, match="duplicate"):
        summarize_trace(runner, dup, summary, runner_sha256="fixed-sha")


def test_rejects_inconsistent_final_statistics() -> None:
    runner, details, summary = fixtures()
    summary["conditions"]["gated_low_destabilize"]["correct_tasks"] = 1
    with pytest.raises(ValueError, match="does not reconcile"):
        summarize_trace(runner, details, summary, runner_sha256="fixed-sha")
