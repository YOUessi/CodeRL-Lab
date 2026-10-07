import pytest

from coderl_lab.analysis.success_concentration import (
    summarize_success_concentration,
)


def test_success_concentration() -> None:
    summary = {
        "per_task": {
            "a": {"samples": 4, "correct": 4},
            "b": {"samples": 4, "correct": 2},
            "c": {"samples": 4, "correct": 0},
        }
    }
    out = summarize_success_concentration(summary)
    assert out["tasks"] == 3
    assert out["total_correct_candidates"] == 6
    assert out["zero_correct_tasks"] == 1
    assert out["all_samples_correct_tasks"] == 1
    assert out["hhi_success_mass"] == pytest.approx((4/6)**2 + (2/6)**2)
