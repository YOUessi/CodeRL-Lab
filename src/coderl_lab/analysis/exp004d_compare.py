from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


PASS_KEYS = ("pass@1", "pass@4", "pass@8", "pass@16")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def summarize(
    *,
    sft_eval: dict[str, Any],
    sft_concentration: dict[str, Any],
    arm_dirs: dict[str, Path],
) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for name, root in arm_dirs.items():
        eval_summary = _load(root / "evaluation" / "enhanced_summary.json")
        analysis_root = root.parents[1] / "analysis" / name
        behavior = _load(analysis_root / "behavior_vs_sft.json")
        concentration = _load(analysis_root / "success_concentration.json")
        bootstrap = _load(analysis_root / "passk_vs_sft.json")

        deltas = {
            key: (
                float(eval_summary["pass_at_k"][key])
                - float(sft_eval["pass_at_k"][key])
            )
            for key in PASS_KEYS
        }
        arms[name] = {
            "pass_at_k": dict(eval_summary["pass_at_k"]),
            "pass_at_k_delta_vs_sft": deltas,
            "solved_tasks": int(eval_summary["solved_tasks"]),
            "solved_delta_vs_sft": (
                int(eval_summary["solved_tasks"]) - int(sft_eval["solved_tasks"])
            ),
            "hidden_mean": float(eval_summary["hidden_mean_pass_rate"]),
            "hidden_mean_delta_vs_sft": (
                float(eval_summary["hidden_mean_pass_rate"])
                - float(sft_eval["hidden_mean_pass_rate"])
            ),
            "exact_completion_match_fraction_vs_sft": float(
                behavior["exact_completion_match_fraction"]
            ),
            "changed_rows_vs_sft": int(behavior["changed_rows"]),
            "changed_tasks_vs_sft": int(behavior["changed_tasks"]),
            "diversity_delta_vs_sft": dict(
                behavior["candidate_minus_reference_diversity"]
            ),
            "success_hhi": float(concentration["hhi_success_mass"]),
            "effective_task_count": float(
                concentration["effective_task_count_from_hhi"]
            ),
            "success_hhi_delta_vs_sft": (
                float(concentration["hhi_success_mass"])
                - float(sft_concentration["hhi_success_mass"])
            ),
            "effective_task_count_delta_vs_sft": (
                float(concentration["effective_task_count_from_hhi"])
                - float(sft_concentration["effective_task_count_from_hhi"])
            ),
            "bootstrap_vs_sft": bootstrap,
        }

    aggregate: dict[str, Any] = {
        "num_seeds": len(arms),
        "pass_at_k_delta_vs_sft": {},
    }
    for key in PASS_KEYS:
        values = [
            float(row["pass_at_k_delta_vs_sft"][key])
            for row in arms.values()
        ]
        aggregate["pass_at_k_delta_vs_sft"][key] = {
            **_stats(values),
            "positive_seed_count": sum(value > 0 for value in values),
            "nonnegative_seed_count": sum(value >= 0 for value in values),
        }

    solved_deltas = [
        float(row["solved_delta_vs_sft"]) for row in arms.values()
    ]
    hhi_deltas = [
        float(row["success_hhi_delta_vs_sft"]) for row in arms.values()
    ]
    effective_deltas = [
        float(row["effective_task_count_delta_vs_sft"])
        for row in arms.values()
    ]
    exact_matches = [
        float(row["exact_completion_match_fraction_vs_sft"])
        for row in arms.values()
    ]

    aggregate["solved_delta_vs_sft"] = _stats(solved_deltas)
    aggregate["success_hhi_delta_vs_sft"] = _stats(hhi_deltas)
    aggregate["effective_task_count_delta_vs_sft"] = _stats(effective_deltas)
    aggregate["exact_completion_match_fraction_vs_sft"] = _stats(exact_matches)

    return {
        "sft_reference": {
            "pass_at_k": dict(sft_eval["pass_at_k"]),
            "solved_tasks": int(sft_eval["solved_tasks"]),
            "hidden_mean": float(sft_eval["hidden_mean_pass_rate"]),
            "success_hhi": float(sft_concentration["hhi_success_mass"]),
            "effective_task_count": float(
                sft_concentration["effective_task_count_from_hhi"]
            ),
        },
        "arms": arms,
        "aggregate": aggregate,
    }


def parse_arm(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("arm must be NAME=DIR")
    name, path = value.split("=", 1)
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-eval", type=Path, required=True)
    parser.add_argument("--sft-concentration", type=Path, required=True)
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = summarize(
        sft_eval=_load(args.sft_eval),
        sft_concentration=_load(args.sft_concentration),
        arm_dirs=dict(parse_arm(value) for value in args.arm),
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
