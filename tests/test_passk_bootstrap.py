import pytest

from coderl_lab.analysis.passk_bootstrap import pass_at_k, task_passk


def test_pass_at_k_edges() -> None:
    assert pass_at_k(16, 0, 4) == 0.0
    assert pass_at_k(16, 16, 4) == 1.0
    assert pass_at_k(16, 1, 16) == 1.0


def test_task_passk() -> None:
    summary = {
        "per_task": {
            "a": {"samples": 16, "correct": 1},
            "b": {"samples": 16, "correct": 0},
        }
    }
    out = task_passk(summary, 4)
    assert out["b"] == 0.0
    assert out["a"] == pytest.approx(0.25)
