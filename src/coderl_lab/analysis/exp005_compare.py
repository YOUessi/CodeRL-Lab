from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def arm_metrics(
    *,
    training: dict[str, Any],
    dynamics: dict[str, Any],
    evaluation: dict[str, Any],
    rl_completions: int,
) -> dict[str, Any]:
    steps = int(dynamics["num_logged_steps"])
    zero_grad_steps = int(dynamics["zero_grad_steps"])
    effective_steps = steps - zero_grad_steps
    return {
        "optimizer_steps": steps,
        "train_runtime_seconds": float(training["metrics"]["train_runtime"]),
        "reward_mean": float(dynamics["overall_mean"]["reward"]),
        "reward_std_mean": float(dynamics["overall_mean"]["reward_std"]),
        "frac_reward_zero_std_mean": float(
            dynamics["overall_mean"]["frac_reward_zero_std"]
        ),
        "zero_grad_steps": zero_grad_steps,
        "zero_grad_fraction": float(dynamics["zero_grad_fraction"]),
        "effective_steps": effective_steps,
        "rl_completions": rl_completions,
        "rl_completions_per_effective_step": (
            rl_completions / effective_steps
            if effective_steps > 0
            else None
        ),
        "entropy_mean": float(dynamics["overall_mean"]["entropy"]),
        "completion_mean_length": float(
            dynamics["overall_mean"]["completions/mean_length"]
        ),
        "pass_at_1": float(evaluation["pass_at_k"]["pass@1"]),
        "pass_at_4": float(evaluation["pass_at_k"]["pass@4"]),
        "hidden_mean_pass_rate": float(
            evaluation["hidden_mean_pass_rate"]
        ),
        "solved_tasks": int(evaluation["solved_tasks"]),
        "all_samples_correct_tasks": int(
            evaluation["all_samples_correct_tasks"]
        ),
        "syntax_failures": int(evaluation["syntax_failures"]),
    }


def delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "train_runtime_seconds",
        "reward_mean",
        "frac_reward_zero_std_mean",
        "zero_grad_fraction",
        "effective_steps",
        "rl_completions_per_effective_step",
        "entropy_mean",
        "pass_at_1",
        "pass_at_4",
        "hidden_mean_pass_rate",
        "solved_tasks",
        "all_samples_correct_tasks",
        "syntax_failures",
    )
    out: dict[str, Any] = {}
    for key in keys:
        av = a.get(key)
        bv = b.get(key)
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            out[key] = bv - av
        else:
            out[key] = None
    return out


def compare_arms(
    *,
    full: dict[str, Any],
    random_control: dict[str, Any],
    boundary: dict[str, Any],
    screening_completions: int,
) -> dict[str, Any]:
    return {
        "arms": {
            "full_random": full,
            "random_subset": random_control,
            "boundary_subset": boundary,
        },
        "primary_delta_boundary_minus_random_subset": delta(
            random_control, boundary
        ),
        "secondary_delta_boundary_minus_full_random": delta(
            full, boundary
        ),
        "screening_overhead": {
            "completions": screening_completions,
            "note": (
                "Boundary arm has this additional offline screening cost. "
                "RL training budgets are matched across arms."
            ),
        },
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare EXP-003 / EXP-005A matched-budget arms"
    )
    for arm in ("full", "random", "boundary"):
        p.add_argument(f"--{arm}-training", type=Path, required=True)
        p.add_argument(f"--{arm}-dynamics", type=Path, required=True)
        p.add_argument(f"--{arm}-evaluation", type=Path, required=True)
    p.add_argument("--rl-completions", type=int, default=1496)
    p.add_argument("--screening-completions", type=int, default=1496)
    p.add_argument("--output", type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    arms = {}
    for name in ("full", "random", "boundary"):
        arms[name] = arm_metrics(
            training=_load(getattr(args, f"{name}_training")),
            dynamics=_load(getattr(args, f"{name}_dynamics")),
            evaluation=_load(getattr(args, f"{name}_evaluation")),
            rl_completions=args.rl_completions,
        )

    result = compare_arms(
        full=arms["full"],
        random_control=arms["random"],
        boundary=arms["boundary"],
        screening_completions=args.screening_completions,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
