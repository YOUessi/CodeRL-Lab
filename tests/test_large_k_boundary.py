import pytest

from coderl_lab.analysis.large_k_boundary import compare_large_k


def summary(rows):
    return {
        "pass_at_k": {
            "pass@1": 0.1,
            "pass@4": 0.2,
            "pass@8": 0.3,
            "pass@16": 0.4,
        },
        "per_task": {
            task_id: {
                "samples": 16,
                "correct": correct,
                "empirical_success_rate": correct / 16,
            }
            for task_id, correct in rows.items()
        },
    }


def test_large_k_support_tracking() -> None:
    base = summary({"a": 2, "b": 1, "c": 0, "d": 16})
    sft = summary({"a": 4, "b": 0, "c": 3, "d": 16})
    grpo = summary({"a": 5, "b": 1, "c": 0, "d": 16})

    out = compare_large_k(base=base, sft=sft, grpo=grpo)

    assert out["support_sets"]["base_lost_by_sft"] == 1
    assert out["support_sets"]["sft_new_vs_base"] == 1
    assert out["support_sets"]["base_lost_then_recovered_by_grpo"] == 1
    assert out["support_sets"]["recovery_fraction_grpo"] == pytest.approx(1.0)
    assert out["task_ids"]["base_lost_by_sft"] == ["b"]
    assert out["task_ids"]["sft_new_vs_base"] == ["c"]
    assert out["task_ids"]["grpo_new_vs_sft"] == ["b"]
