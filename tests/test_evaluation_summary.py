import pytest

from coderl_lab.analysis.evaluation_summary import summarize_evaluation


def test_summarize_evaluation() -> None:
    details = [
        {
            "public": {"syntax_ok": True, "pass_rate": 1.0},
            "hidden": {"pass_rate": 1.0},
            "hidden_all_pass": True,
        },
        {
            "public": {"syntax_ok": True, "pass_rate": 1.0},
            "hidden": {"pass_rate": 0.5},
            "hidden_all_pass": False,
        },
        {
            "public": {"syntax_ok": False, "pass_rate": 0.0},
            "hidden": {"pass_rate": 0.0},
            "hidden_all_pass": False,
        },
    ]
    summary = {
        "num_tasks": 2,
        "num_predictions": 3,
        "pass_at_k": {"pass@1": 0.25},
        "per_task": {
            "a": {"samples": 2, "correct": 1},
            "b": {"samples": 1, "correct": 0},
        },
    }
    out = summarize_evaluation(details, summary)
    assert out["syntax_failures"] == 1
    assert out["syntax_failure_rate"] == pytest.approx(1 / 3)
    assert out["public_mean_pass_rate"] == pytest.approx(2 / 3)
    assert out["hidden_mean_pass_rate"] == pytest.approx(0.5)
    assert out["public_all_hidden_fail_candidates"] == 1
    assert out["solved_tasks"] == 1
