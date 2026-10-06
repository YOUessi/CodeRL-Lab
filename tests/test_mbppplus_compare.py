import pytest

from coderl_lab.analysis.mbppplus_compare import compare


def summary(a, b):
    per_task = {
        "x": {
            "base_correct": a[0],
            "plus_correct": b[0],
            "base_success_rate": a[0] / 2,
            "plus_success_rate": b[0] / 2,
        },
        "y": {
            "base_correct": a[1],
            "plus_correct": b[1],
            "base_success_rate": a[1] / 2,
            "plus_success_rate": b[1] / 2,
        },
    }
    return {
        "base_pass_at_k": {"pass@1": sum(a) / 4},
        "plus_pass_at_k": {"pass@1": sum(b) / 4},
        "base_solved_tasks": sum(x > 0 for x in a),
        "plus_solved_tasks": sum(x > 0 for x in b),
        "mean_base_success_rate": sum(x / 2 for x in a) / 2,
        "mean_plus_success_rate": sum(x / 2 for x in b) / 2,
        "mean_robustness_gap": sum((x-y) / 2 for x, y in zip(a, b)) / 2,
        "per_task": per_task,
    }


def test_compare_mbppplus_models() -> None:
    base = summary([1, 1], [1, 0])
    sft = summary([2, 1], [2, 1])
    grpo = summary([2, 2], [2, 1])
    out = compare(base=base, sft=sft, grpo=grpo, iterations=1000, seed=7)
    assert out["models"]["base"]["candidate_robust_retention"] == pytest.approx(0.5)
    assert out["models"]["sft"]["candidate_robust_retention"] == pytest.approx(1.0)
    assert (
        out["paired_bootstrap"]["sft_minus_base"]["plus_empirical_success_rate"][
            "observed_delta"
        ]
        == pytest.approx(0.5)
    )
