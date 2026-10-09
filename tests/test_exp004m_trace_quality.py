from __future__ import annotations

import copy

import pytest

from coderl_lab.analysis.exp004m_trace_quality import summarize_quality


def fixture_runner() -> dict:
    records = {}
    for i in range(175):
        records[str(i)] = {
            "platform": "atcoder" if i < 112 else "leetcode",
            "difficulty": "hard" if i < 80 else "easy",
            "public_all_pass": i != 0,
            "gate_triggered": i == 0,
            "reference_token_count": 512 if i == 0 else 15,
            "baseline": {"code": "def f(: pass" if i == 0 else "print(1)"},
            "gated_low_destabilize": {
                "code": "print(1)" if i == 0 else "print(1)"
            },
        }
    return {
        "experiment": "EXP-004L", "private_tests_accessed": False,
        "max_new_tokens": 512, "tasks": 175, "per_task": records,
    }


def test_aggregated_quality_audit_has_no_code_or_hidden_test_dump() -> None:
    x = summarize_quality(
        fixture_runner(),
        expected_runner_sha256="correct",
        actual_runner_sha256="correct",
    )
    assert x["overall"]["tasks"] == 175
    assert x["overall"]["hit_512_token_cap"] == 1
    assert x["overall"]["baseline_unparseable"] == 1
    assert x["overall"]["gated_unparseable"] == 0
    assert x["overall"]["unparseable_at_cap"] == 1
    assert x["groups"]["gated"]["true"]["changed"] == 1
    assert x["groups"]["difficulty"]["hard"]["tasks"] == 80
    assert x["no_private_test_data_or_raw_code_exported"] is True
    assert "def f(" not in str(x)


def test_quality_requires_frozen_complete_runner_and_hash() -> None:
    runner = fixture_runner()
    with pytest.raises(ValueError, match="checksum"):
        summarize_quality(runner, expected_runner_sha256="a", actual_runner_sha256="b")
    runner["max_new_tokens"] = 1024
    with pytest.raises(ValueError, match="token cap"):
        summarize_quality(runner, expected_runner_sha256="a", actual_runner_sha256="a")


def test_out_of_range_generated_length_fails_closed() -> None:
    runner = fixture_runner()
    runner["per_task"]["0"]["reference_token_count"] = 513
    with pytest.raises(ValueError, match="token length"):
        summarize_quality(runner, expected_runner_sha256="a", actual_runner_sha256="a")
