import pytest
from coderl_lab.analysis.exp7c_sandbox_score import (
    bootstrap_deltas, image_sha, sandbox_check,
)


def test_pinned_sandbox_passes_synthetic_function():
    assert image_sha().startswith("sha256:")
    result=sandbox_check(
        "def plus_one(x):\n    return x + 1",
        "def check(candidate):\n    assert candidate(2) == 3",
        "plus_one",
    )
    assert result == (True, "none")


def test_pinned_sandbox_rejects_incorrect_answer():
    result=sandbox_check(
        "def plus_one(x):\n    return 5",
        "def check(candidate):\n    assert candidate(2) == 3",
        "plus_one",
    )
    assert result == (False, "AssertionError")


def test_paired_ci_synthetic_identity():
    paired=[{"base":True,"sft":True} for _ in range(164)]
    outcome=bootstrap_deltas(paired,iterations=100,seed=42)
    assert outcome["observed_delta"]==0.0
    assert outcome["ci95_low"]==0.0
    assert outcome["ci95_high"]==0.0
