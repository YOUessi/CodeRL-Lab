from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def online_arm_metrics(
    *,
    training: dict[str, Any],
    dynamics: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    online = dict(training["online_sampling"])
    state = dict(online["state_summary"])

    steps = int(dynamics["num_logged_steps"])
    zero_grad_steps = int(dynamics["zero_grad_steps"])
    effective_steps = steps - zero_grad_steps
    actual_completions = int(online["actual_rollout_completions"])
    groups_observed = int(online["actual_rollout_groups"])

    return {
        "optimizer_steps": steps,
        "train_runtime_seconds": float(training["metrics"]["train_runtime"]),
        "actual_rollout_groups": groups_observed,
        "actual_rl_completions": actual_completions,
        "prefetched_unobserved_group_selections": int(
            online["prefetched_unobserved_group_selections"]
        ),
        "reward_mean": float(dynamics["overall_mean"]["reward"]),
        "frac_reward_zero_std_mean": float(
            dynamics["overall_mean"]["frac_reward_zero_std"]
        ),
        "zero_grad_steps": zero_grad_steps,
        "zero_grad_fraction": float(dynamics["zero_grad_fraction"]),
        "effective_steps": effective_steps,
        "completions_per_effective_step": (
            actual_completions / effective_steps
            if effective_steps > 0
            else None
        ),
        "entropy_mean": float(dynamics["overall_mean"]["entropy"]),
        "pass_at_1": float(evaluation["pass_at_k"]["pass@1"]),
        "pass_at_4": float(evaluation["pass_at_k"]["pass@4"]),
        "hidden_mean": float(evaluation["hidden_mean_pass_rate"]),
        "solved_tasks": int(evaluation["solved_tasks"]),
        "all4_tasks": int(evaluation["all_samples_correct_tasks"]),
        "syntax_failures": int(evaluation["syntax_failures"]),
        "online_state": {
            "status_counts": dict(state["status_counts"]),
            "groups_observed": int(state["groups_observed"]),
            "mixed_groups_observed": int(state["mixed_groups_observed"]),
            "flat_groups_observed": int(state["flat_groups_observed"]),
            "observed_mixed_fraction": float(
                state["observed_mixed_fraction"]
            ),
            "total_group_selections": int(state["total_group_selections"]),
            "prefetched_unobserved_group_selections": int(
                state["prefetched_unobserved_group_selections"]
            ),
            "exploit_selections": int(state["exploit_selections"]),
            "explore_selections": int(state["explore_selections"]),
            "unique_selected_tasks": int(state["unique_selected_tasks"]),
            "unique_observed_tasks": int(state["unique_observed_tasks"]),
            "transition_counts": dict(state["transition_counts"]),
        },
    }


def delta(base: dict[str, Any], online: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "train_runtime_seconds",
        "frac_reward_zero_std",
        "zero_grad_fraction",
        "effective_steps",
        "completions_per_effective_step",
        "pass_at_1",
        "pass_at_4",
        "hidden_mean",
        "solved_tasks",
        "all4_tasks",
    )
    out: dict[str, Any] = {}
    for key in keys:
        base_value = base.get(key)
        online_key = (
            "frac_reward_zero_std_mean"
            if key == "frac_reward_zero_std"
            else key
        )
        online_value = online.get(online_key)
        if isinstance(base_value, (int, float)) and isinstance(
            online_value, (int, float)
        ):
            out[key] = online_value - base_value
        else:
            out[key] = None
    return out


def compare(
    *,
    exp005a: dict[str, Any],
    online: dict[str, Any],
) -> dict[str, Any]:
    full = dict(exp005a["arms"]["full_random"])
    random_subset = dict(exp005a["arms"]["random_subset"])
    boundary = dict(exp005a["arms"]["boundary_subset"])

    return {
        "baselines": {
            "full_random": full,
            "random_subset": random_subset,
            "static_boundary": boundary,
        },
        "online_dynamic": online,
        "delta_online_minus_full_random": delta(full, online),
        "delta_online_minus_random_subset": delta(random_subset, online),
        "delta_online_minus_static_boundary": delta(boundary, online),
        "budget_note": (
            "EXP-005B uses no offline screening. Compare actual observed "
            "rollout completions, not sampler prefetch selections."
        ),
    }


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare EXP-005B online sampler with prior baselines"
    )
    p.add_argument("--exp005a-summary", type=Path, required=True)
    p.add_argument("--online-training", type=Path, required=True)
    p.add_argument("--online-dynamics", type=Path, required=True)
    p.add_argument("--online-evaluation", type=Path, required=True)
    p.add_argument("--output", type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    online = online_arm_metrics(
        training=load(args.online_training),
        dynamics=load(args.online_dynamics),
        evaluation=load(args.online_evaluation),
    )
    result = compare(
        exp005a=load(args.exp005a_summary),
        online=online,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
