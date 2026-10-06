from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare(
    *,
    baseline_dynamics: dict[str, Any],
    process_dynamics: dict[str, Any],
    baseline_eval: dict[str, Any],
    process_eval: dict[str, Any],
    baseline_diag: dict[str, Any],
    process_diag: dict[str, Any],
) -> dict[str, Any]:
    old = baseline_dynamics["overall_mean"]
    new = process_dynamics["overall_mean"]

    training = {
        "baseline_zero_grad_fraction": baseline_dynamics["zero_grad_fraction"],
        "process_zero_grad_fraction": process_dynamics["zero_grad_fraction"],
        "zero_grad_fraction_delta": (
            process_dynamics["zero_grad_fraction"]
            - baseline_dynamics["zero_grad_fraction"]
        ),
        "baseline_frac_reward_zero_std": old["frac_reward_zero_std"],
        "process_frac_reward_zero_std": new["frac_reward_zero_std"],
        "frac_reward_zero_std_delta": (
            new["frac_reward_zero_std"] - old["frac_reward_zero_std"]
        ),
        "baseline_public_reward_mean": old["reward/public_mean"],
        "process_public_reward_mean": new["reward/public_mean"],
        "public_reward_mean_delta": (
            new["reward/public_mean"] - old["reward/public_mean"]
        ),
        "process_dependency_complete_mean": new.get(
            "reward/dependency_complete_mean"
        ),
        "process_runtime_clean_mean": new.get("reward/runtime_clean_mean"),
        "baseline_entropy_mean": old["entropy"],
        "process_entropy_mean": new["entropy"],
    }

    keys = [
        "pass_at_k",
        "hidden_mean_pass_rate",
        "solved_tasks",
        "all_samples_correct_tasks",
        "syntax_failures",
    ]
    validation = {
        "baseline": {k: baseline_eval[k] for k in keys},
        "process": {k: process_eval[k] for k in keys},
        "delta": {
            "pass@1": (
                process_eval["pass_at_k"]["pass@1"]
                - baseline_eval["pass_at_k"]["pass@1"]
            ),
            "pass@4": (
                process_eval["pass_at_k"]["pass@4"]
                - baseline_eval["pass_at_k"]["pass@4"]
            ),
            "pass@8": (
                process_eval["pass_at_k"]["pass@8"]
                - baseline_eval["pass_at_k"]["pass@8"]
            ),
            "pass@16": (
                process_eval["pass_at_k"]["pass@16"]
                - baseline_eval["pass_at_k"]["pass@16"]
            ),
            "hidden_mean_pass_rate": (
                process_eval["hidden_mean_pass_rate"]
                - baseline_eval["hidden_mean_pass_rate"]
            ),
            "solved_tasks": (
                process_eval["solved_tasks"] - baseline_eval["solved_tasks"]
            ),
            "syntax_failures": (
                process_eval["syntax_failures"]
                - baseline_eval["syntax_failures"]
            ),
        },
    }

    diagnostics = {
        "baseline": {
            "dependency_incomplete_count": baseline_diag[
                "dependency_incomplete_count"
            ],
            "runtime_unclean_count": baseline_diag["runtime_unclean_count"],
            "top_unresolved_names": baseline_diag["top_unresolved_names"],
            "runtime_failure_types": baseline_diag["runtime_failure_types"],
        },
        "process": {
            "dependency_incomplete_count": process_diag[
                "dependency_incomplete_count"
            ],
            "runtime_unclean_count": process_diag["runtime_unclean_count"],
            "top_unresolved_names": process_diag["top_unresolved_names"],
            "runtime_failure_types": process_diag["runtime_failure_types"],
        },
        "delta": {
            "dependency_incomplete_count": (
                process_diag["dependency_incomplete_count"]
                - baseline_diag["dependency_incomplete_count"]
            ),
            "runtime_unclean_count": (
                process_diag["runtime_unclean_count"]
                - baseline_diag["runtime_unclean_count"]
            ),
        },
    }

    return {
        "training": training,
        "validation": validation,
        "diagnostics": diagnostics,
        "note": (
            "Total reward means are intentionally not compared because "
            "EXP-006A and EXP-004A use different reward scales."
        ),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare outcome and process GRPO")
    p.add_argument("--baseline-dynamics", type=Path, required=True)
    p.add_argument("--process-dynamics", type=Path, required=True)
    p.add_argument("--baseline-eval", type=Path, required=True)
    p.add_argument("--process-eval", type=Path, required=True)
    p.add_argument("--baseline-diagnostics", type=Path, required=True)
    p.add_argument("--process-diagnostics", type=Path, required=True)
    p.add_argument("--output", type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = compare(
        baseline_dynamics=_load(args.baseline_dynamics),
        process_dynamics=_load(args.process_dynamics),
        baseline_eval=_load(args.baseline_eval),
        process_eval=_load(args.process_eval),
        baseline_diag=_load(args.baseline_diagnostics),
        process_diag=_load(args.process_diagnostics),
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
