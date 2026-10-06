from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _delta(after: dict[str, Any], before: dict[str, Any], key: str) -> float:
    return float(after[key]) - float(before[key])


def summarize_scale(
    *,
    base: dict[str, Any],
    sft: dict[str, Any],
    grpo: dict[str, Any],
    grpo_dynamics: dict[str, Any],
) -> dict[str, Any]:
    out = {
        "base": base,
        "sft": sft,
        "grpo": grpo,
        "sft_minus_base": {
            "pass_at_1": _delta(sft, base, "pass_at_1"),
            "pass_at_4": _delta(sft, base, "pass_at_4"),
            "hidden_mean": _delta(sft, base, "hidden_mean"),
            "solved_tasks": int(sft["solved_tasks"]) - int(base["solved_tasks"]),
        },
        "grpo_minus_sft": {
            "pass_at_1": _delta(grpo, sft, "pass_at_1"),
            "pass_at_4": _delta(grpo, sft, "pass_at_4"),
            "hidden_mean": _delta(grpo, sft, "hidden_mean"),
            "solved_tasks": int(grpo["solved_tasks"]) - int(sft["solved_tasks"]),
        },
        "grpo_training": {
            "zero_grad_fraction": float(grpo_dynamics["zero_grad_fraction"]),
            "frac_reward_zero_std": float(
                grpo_dynamics["overall_mean"]["frac_reward_zero_std"]
            ),
            "reward_mean": float(grpo_dynamics["overall_mean"]["reward"]),
            "entropy_mean": float(grpo_dynamics["overall_mean"]["entropy"]),
            "effective_steps": (
                int(grpo_dynamics["num_logged_steps"])
                - int(grpo_dynamics["zero_grad_steps"])
            ),
        },
    }
    return out


def compare_scales(
    small: dict[str, Any],
    large: dict[str, Any],
) -> dict[str, Any]:
    return {
        "small_model": small,
        "large_model": large,
        "cross_scale": {
            "base_pass_at_1": large["base"]["pass_at_1"] - small["base"]["pass_at_1"],
            "base_pass_at_4": large["base"]["pass_at_4"] - small["base"]["pass_at_4"],
            "sft_gain_pass_at_1": (
                large["sft_minus_base"]["pass_at_1"]
                - small["sft_minus_base"]["pass_at_1"]
            ),
            "sft_gain_pass_at_4": (
                large["sft_minus_base"]["pass_at_4"]
                - small["sft_minus_base"]["pass_at_4"]
            ),
            "grpo_gain_pass_at_1": (
                large["grpo_minus_sft"]["pass_at_1"]
                - small["grpo_minus_sft"]["pass_at_1"]
            ),
            "grpo_gain_pass_at_4": (
                large["grpo_minus_sft"]["pass_at_4"]
                - small["grpo_minus_sft"]["pass_at_4"]
            ),
            "grpo_zero_grad_fraction": (
                large["grpo_training"]["zero_grad_fraction"]
                - small["grpo_training"]["zero_grad_fraction"]
            ),
            "grpo_frac_reward_zero_std": (
                large["grpo_training"]["frac_reward_zero_std"]
                - small["grpo_training"]["frac_reward_zero_std"]
            ),
        },
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare 0.6B and 1.7B CodeRL-Lab baselines")
    p.add_argument("--small-summary", type=Path, required=True)
    p.add_argument("--large-summary", type=Path, required=True)
    p.add_argument("--output", type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    small = _load(args.small_summary)
    large = _load(args.large_summary)
    result = compare_scales(small, large)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
