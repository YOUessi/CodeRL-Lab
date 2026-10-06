import pytest

from coderl_lab.analysis.grpo_history import summarize_history


def test_summarize_grpo_history() -> None:
    history = [
        {
            "reward": 0.1,
            "reward_std": 0.0,
            "frac_reward_zero_std": 1.0,
            "entropy": 0.4,
            "grad_norm": 0.0,
            "completions/mean_length": 50.0,
            "step_time": 2.0,
            "reward/public_mean": 0.0,
            "reward/syntax_mean": 1.0,
            "reward/public_all_pass_mean": 0.0,
        },
        {
            "reward": 0.5,
            "reward_std": 0.4,
            "frac_reward_zero_std": 0.5,
            "entropy": 0.3,
            "grad_norm": 1.2,
            "completions/mean_length": 40.0,
            "step_time": 3.0,
            "reward/public_mean": 0.5,
            "reward/syntax_mean": 1.0,
            "reward/public_all_pass_mean": 0.5,
        },
        {
            "reward": 0.9,
            "reward_std": 0.2,
            "frac_reward_zero_std": 0.0,
            "entropy": 0.2,
            "grad_norm": 0.8,
            "completions/mean_length": 30.0,
            "step_time": 2.5,
            "reward/public_mean": 0.875,
            "reward/syntax_mean": 1.0,
            "reward/public_all_pass_mean": 0.75,
        },
        {"train_runtime": 10.0},
    ]
    summary = summarize_history(history)
    assert summary["num_logged_steps"] == 3
    assert summary["overall_mean"]["reward"] == pytest.approx(0.5)
    assert summary["zero_grad_steps"] == 1
    assert summary["zero_reward_std_steps"] == 1
    assert summary["first_quarter_mean"]["reward"] == pytest.approx(0.1)
    assert summary["last_quarter_mean"]["reward"] == pytest.approx(0.9)
