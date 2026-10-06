import pytest

from coderl_lab.analysis.scale_compare import compare_scales, summarize_scale


def metric(p1, p4, h, solved):
    return {
        "pass_at_1": p1,
        "pass_at_4": p4,
        "hidden_mean": h,
        "solved_tasks": solved,
    }


def dynamics(zero_grad, zero_std):
    return {
        "num_logged_steps": 10,
        "zero_grad_steps": int(zero_grad * 10),
        "zero_grad_fraction": zero_grad,
        "overall_mean": {
            "frac_reward_zero_std": zero_std,
            "reward": 0.5,
            "entropy": 0.2,
        },
    }


def test_summarize_scale() -> None:
    out = summarize_scale(
        base=metric(0.1, 0.2, 0.1, 2),
        sft=metric(0.2, 0.3, 0.2, 3),
        grpo=metric(0.25, 0.28, 0.22, 3),
        grpo_dynamics=dynamics(0.2, 0.4),
    )
    assert out["sft_minus_base"]["pass_at_1"] == pytest.approx(0.1)
    assert out["grpo_minus_sft"]["pass_at_4"] == pytest.approx(-0.02)
    assert out["grpo_training"]["effective_steps"] == 8


def test_compare_scales() -> None:
    small = summarize_scale(
        base=metric(0.1, 0.2, 0.1, 2),
        sft=metric(0.2, 0.3, 0.2, 3),
        grpo=metric(0.22, 0.29, 0.21, 3),
        grpo_dynamics=dynamics(0.3, 0.5),
    )
    large = summarize_scale(
        base=metric(0.2, 0.4, 0.2, 4),
        sft=metric(0.3, 0.5, 0.3, 5),
        grpo=metric(0.35, 0.48, 0.32, 5),
        grpo_dynamics=dynamics(0.1, 0.3),
    )
    out = compare_scales(small, large)
    assert out["cross_scale"]["base_pass_at_1"] == pytest.approx(0.1)
    assert out["cross_scale"]["grpo_gain_pass_at_1"] == pytest.approx(0.03)
    assert out["cross_scale"]["grpo_zero_grad_fraction"] == pytest.approx(-0.2)
