import pytest

from coderl_lab.analysis.exp005_compare import (
    arm_metrics,
    compare_arms,
)


def make_arm(zero_grad: int, p1: float, p4: float) -> dict:
    training = {"metrics": {"train_runtime": 100.0}}
    dynamics = {
        "num_logged_steps": 10,
        "zero_grad_steps": zero_grad,
        "zero_grad_fraction": zero_grad / 10,
        "overall_mean": {
            "reward": 0.5,
            "reward_std": 0.3,
            "frac_reward_zero_std": 0.4,
            "entropy": 0.2,
            "completions/mean_length": 50.0,
        },
    }
    evaluation = {
        "pass_at_k": {"pass@1": p1, "pass@4": p4},
        "hidden_mean_pass_rate": p1,
        "solved_tasks": 4,
        "all_samples_correct_tasks": 1,
        "syntax_failures": 0,
    }
    return arm_metrics(
        training=training,
        dynamics=dynamics,
        evaluation=evaluation,
        rl_completions=80,
    )


def test_effective_rollout_efficiency() -> None:
    arm = make_arm(2, 0.2, 0.4)
    assert arm["effective_steps"] == 8
    assert arm["rl_completions_per_effective_step"] == pytest.approx(10.0)


def test_boundary_primary_delta_is_against_same_size_control() -> None:
    full = make_arm(3, 0.20, 0.40)
    random = make_arm(4, 0.21, 0.39)
    boundary = make_arm(1, 0.25, 0.42)
    out = compare_arms(
        full=full,
        random_control=random,
        boundary=boundary,
        screening_completions=80,
    )
    d = out["primary_delta_boundary_minus_random_subset"]
    assert d["zero_grad_fraction"] == pytest.approx(-0.3)
    assert d["pass_at_1"] == pytest.approx(0.04)
    assert out["screening_overhead"]["completions"] == 80
