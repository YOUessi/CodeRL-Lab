import pytest

from coderl_lab.analysis.exp005b_compare import (
    compare,
    online_arm_metrics,
)


def test_online_arm_uses_actual_rollouts_not_prefetch() -> None:
    training = {
        "metrics": {"train_runtime": 10.0},
        "online_sampling": {
            "actual_rollout_groups": 8,
            "actual_rollout_completions": 32,
            "prefetched_unobserved_group_selections": 2,
            "state_summary": {
                "status_counts": {"unknown": 20, "mixed": 5, "flat": 7},
                "groups_observed": 8,
                "mixed_groups_observed": 5,
                "flat_groups_observed": 3,
                "observed_mixed_fraction": 0.625,
                "total_group_selections": 10,
                "prefetched_unobserved_group_selections": 2,
                "exploit_selections": 3,
                "explore_selections": 7,
                "unique_selected_tasks": 7,
                "unique_observed_tasks": 6,
                "transition_counts": {"unknown->mixed": 3},
            },
        },
    }
    dynamics = {
        "num_logged_steps": 4,
        "zero_grad_steps": 1,
        "zero_grad_fraction": 0.25,
        "overall_mean": {
            "reward": 0.5,
            "frac_reward_zero_std": 0.4,
            "entropy": 0.2,
        },
    }
    evaluation = {
        "pass_at_k": {"pass@1": 0.2, "pass@4": 0.4},
        "hidden_mean_pass_rate": 0.22,
        "solved_tasks": 4,
        "all_samples_correct_tasks": 1,
        "syntax_failures": 0,
    }
    out = online_arm_metrics(
        training=training,
        dynamics=dynamics,
        evaluation=evaluation,
    )
    assert out["actual_rl_completions"] == 32
    assert out["completions_per_effective_step"] == pytest.approx(32 / 3)
    assert out["prefetched_unobserved_group_selections"] == 2


def test_compare_uses_online_minus_baseline_direction() -> None:
    exp005a = {
        "arms": {
            "full_random": {
                "train_runtime_seconds": 10.0,
                "frac_reward_zero_std": 0.5,
                "zero_grad_fraction": 0.3,
                "effective_steps": 7,
                "completions_per_effective_step": 12.0,
                "pass_at_1": 0.2,
                "pass_at_4": 0.4,
                "hidden_mean": 0.2,
                "solved_tasks": 4,
                "all4_tasks": 1,
            },
            "random_subset": {
                "train_runtime_seconds": 10.0,
                "frac_reward_zero_std": 0.55,
                "zero_grad_fraction": 0.35,
                "effective_steps": 6,
                "completions_per_effective_step": 13.0,
                "pass_at_1": 0.21,
                "pass_at_4": 0.4,
                "hidden_mean": 0.21,
                "solved_tasks": 4,
                "all4_tasks": 1,
            },
            "boundary_subset": {
                "train_runtime_seconds": 10.0,
                "frac_reward_zero_std": 0.33,
                "zero_grad_fraction": 0.12,
                "effective_steps": 9,
                "completions_per_effective_step": 9.0,
                "pass_at_1": 0.19,
                "pass_at_4": 0.4,
                "hidden_mean": 0.2,
                "solved_tasks": 4,
                "all4_tasks": 1,
            },
        }
    }
    online = {
        "train_runtime_seconds": 9.0,
        "frac_reward_zero_std_mean": 0.25,
        "zero_grad_fraction": 0.1,
        "effective_steps": 10,
        "completions_per_effective_step": 8.0,
        "pass_at_1": 0.23,
        "pass_at_4": 0.42,
        "hidden_mean": 0.23,
        "solved_tasks": 5,
        "all4_tasks": 2,
    }
    result = compare(exp005a=exp005a, online=online)
    d = result["delta_online_minus_full_random"]
    assert d["zero_grad_fraction"] == pytest.approx(-0.2)
    assert d["pass_at_1"] == pytest.approx(0.03)
