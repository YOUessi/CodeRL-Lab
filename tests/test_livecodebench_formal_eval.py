from __future__ import annotations

from collections import Counter

import pytest

from coderl_lab.analysis.livecodebench_formal_eval import (
    summarize,
    transitions,
    validate_frozen_input,
)
from coderl_lab.datasets.livecodebench_smoke import select_smoke_tasks


def test_smoke_selection_is_outcome_blind_and_mixed_format() -> None:
    rows = []
    for platform in ("atcoder", "leetcode"):
        for difficulty in ("easy", "medium", "hard"):
            for index in range(4):
                rows.append({
                    "question_id": f"{platform}_{difficulty}_{index:02d}",
                    "platform": platform,
                    "difficulty": difficulty,
                    "public_test_cases": [{
                        "testtype": "stdin" if platform == "atcoder" else "functional",
                    }],
                })
    selected = select_smoke_tasks(list(reversed(rows)))
    assert len(selected) == 12
    assert Counter((row["platform"], row["difficulty"]) for row in selected) == {
        (platform, difficulty): 2
        for platform in ("atcoder", "leetcode")
        for difficulty in ("easy", "medium", "hard")
    }
    assert set(x["question_id"] for x in selected) == {
        f"{platform}_{difficulty}_{i:02d}"
        for platform in ("atcoder", "leetcode")
        for difficulty in ("easy", "medium", "hard")
        for i in (0, 1)
    }


def test_smoke_selection_fails_on_unsupported_coverage() -> None:
    with pytest.raises(ValueError, match="need two"):
        select_smoke_tasks([{
            "question_id": "a", "platform": "atcoder", "difficulty": "easy",
            "public_test_cases": [{"testtype": "stdin"}],
        }])


def _outcome(*, qid: str, baseline: bool, gated: bool, always: bool, 
             public_pass: bool = False) -> dict:
    return {
        "task_id": qid,
        "platform": "atcoder" if qid.startswith("a") else "leetcode",
        "difficulty": "easy",
        "eligible": True,
        "gate_triggered": not public_pass,
        "public_all_pass": public_pass,
        "baseline_correct": baseline,
        "gated_low_destabilize_correct": gated,
        "gated_high_destabilize_correct": baseline,
        "always_low_destabilize_correct": always,
        "baseline_changed": False,
        "gated_low_destabilize_changed": baseline != gated,
        "gated_high_destabilize_changed": False,
        "always_low_destabilize_changed": baseline != always,
    }


def test_pair_transitions_and_primary_metric() -> None:
    rows = [
        _outcome(qid="a1", baseline=False, gated=True, always=True),
        _outcome(qid="a2", baseline=False, gated=True, always=False),
        _outcome(qid="l1", baseline=True, gated=True, always=False, public_pass=True),
        _outcome(qid="l2", baseline=False, gated=False, always=False),
    ]
    out = summarize(rows, iterations=300, seed=42)
    assert out["tasks"] == 4
    assert out["conditions"]["baseline"]["correct_tasks"] == 1
    assert out["conditions"]["gated_low_destabilize"]["correct_tasks"] == 3
    assert out["conditions"]["gated_low_destabilize"]["delta_vs_baseline"] == 0.5
    assert out["primary_success"] is False  # With n=4, the paired CI still includes zero
    assert out["gate_triggered_tasks"] == 3
    assert transitions(rows, "gated_low_destabilize") == {
        "wrong_to_correct": 2,
        "correct_to_wrong": 0,
        "correct_to_correct": 1,
        "wrong_to_wrong": 1,
    }
    assert out["public_pass_protection"]["gated_low_changed"] == 0
    assert out["posthoc_gate_audit"]["gate_selected_hidden_wrong"] == 3


def test_freeze_rejects_private_test_access_without_looking_at_source(
    tmp_path, monkeypatch
) -> None:
    class FakeRun:
        stdout = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24\n"

    monkeypatch.setattr(
        "coderl_lab.analysis.livecodebench_formal_eval.subprocess.run",
        lambda *args, **kwargs: FakeRun(),
    )
    runner_path = tmp_path / "runner.json"
    runner_path.write_text("{}", encoding="utf-8")
    task = {
        "question_id": "q1",
        "eligible": True,
        "gate_triggered": True,
        "public_all_pass": False,
        "baseline": {"raw_completion": "a", "code": "a"},
        "gated_low_destabilize": {"raw_completion": "b"},
        "gated_high_destabilize": {"raw_completion": "a"},
        "always_low_destabilize": {"raw_completion": "b"},
    }
    runner = {
        "experiment": "EXP-004L",
        "tasks": 12,
        "private_tests_accessed": True,
        "official_livecodebench_commit": FakeRun.stdout.strip(),
        "per_task": {"q1": task},
    }
    with pytest.raises(ValueError, match="private-test isolation"):
        validate_frozen_input(
            runner, {"q1": {}}, {"source_revision": "foo"},
            expected_tasks=12, lcb_repo=tmp_path, runner_path=runner_path,
        )
