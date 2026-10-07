from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


PASS_KEYS = ("pass@1", "pass@4", "pass@8", "pass@16")
SCALE_MAP = {
    "025": 0.25,
    "050": 0.50,
    "100": 1.00,
    "200": 2.00,
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }



def _pass_at_k(n: int, c: int, k: int) -> float:
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _task_passk(summary: dict[str, Any], k: int) -> dict[str, float]:
    return {
        str(task_id): _pass_at_k(
            int(row["samples"]),
            int(row["correct"]),
            k,
        )
        for task_id, row in summary["per_task"].items()
    }


def _cluster_bootstrap(
    diffs: list[float],
    *,
    iterations: int = 20000,
    seed: int = 42,
) -> dict[str, float | int]:
    observed = mean(diffs)
    rng = random.Random(seed)
    n = len(diffs)
    values: list[float] = []
    for _ in range(iterations):
        values.append(
            sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        )
    values.sort()
    return {
        "tasks": n,
        "observed_mean_delta": observed,
        "ci95_low": values[int(0.025 * iterations)],
        "ci95_high": values[min(iterations - 1, int(0.975 * iterations))],
        "bootstrap_probability_positive": (
            sum(value > 0 for value in values) / iterations
        ),
        "iterations": iterations,
        "seed": seed,
    }


def parse_arm(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("arm must be NAME=DIR")
    name, raw = value.split("=", 1)
    if not re.fullmatch(r"scale(?:025|050|100|200)_seed(?:101|202|303)", name):
        raise ValueError(f"invalid arm name: {name}")
    return name, Path(raw)


def scale_and_seed(name: str) -> tuple[float, int]:
    match = re.fullmatch(r"scale(025|050|100|200)_seed(101|202|303)", name)
    if match is None:
        raise ValueError(f"invalid arm name: {name}")
    return SCALE_MAP[match.group(1)], int(match.group(2))


def summarize(
    *,
    sft_eval: dict[str, Any],
    sft_summary: dict[str, Any],
    sft_concentration: dict[str, Any],
    analysis_root: Path,
    arm_dirs: dict[str, Path],
) -> dict[str, Any]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    arms: dict[str, Any] = {}

    for name, eval_dir in sorted(arm_dirs.items()):
        scale, seed = scale_and_seed(name)
        eval_summary = _load(eval_dir / "evaluation" / "enhanced_summary.json")
        raw_summary = _load(eval_dir / "evaluation" / "summary.json")
        analysis_dir = analysis_root / name
        behavior = _load(analysis_dir / "behavior_vs_sft.json")
        concentration = _load(analysis_dir / "success_concentration.json")
        bootstrap = _load(analysis_dir / "passk_vs_sft.json")

        pass_delta = {
            key: float(eval_summary["pass_at_k"][key]) - float(sft_eval["pass_at_k"][key])
            for key in PASS_KEYS
        }
        row = {
            "name": name,
            "scale": scale,
            "seed": seed,
            "pass_at_k": dict(eval_summary["pass_at_k"]),
            "pass_delta_vs_sft": pass_delta,
            "solved_tasks": int(eval_summary["solved_tasks"]),
            "solved_delta_vs_sft": int(eval_summary["solved_tasks"]) - int(sft_eval["solved_tasks"]),
            "hidden_mean": float(eval_summary["hidden_mean_pass_rate"]),
            "hidden_mean_delta_vs_sft": (
                float(eval_summary["hidden_mean_pass_rate"])
                - float(sft_eval["hidden_mean_pass_rate"])
            ),
            "exact_match_fraction_vs_sft": float(
                behavior["exact_completion_match_fraction"]
            ),
            "changed_rows_vs_sft": int(behavior["changed_rows"]),
            "changed_tasks_vs_sft": int(behavior["changed_tasks"]),
            "diversity_delta_vs_sft": dict(
                behavior["candidate_minus_reference_diversity"]
            ),
            "success_hhi": float(concentration["hhi_success_mass"]),
            "success_hhi_delta_vs_sft": (
                float(concentration["hhi_success_mass"])
                - float(sft_concentration["hhi_success_mass"])
            ),
            "effective_task_count": float(
                concentration["effective_task_count_from_hhi"]
            ),
            "effective_task_count_delta_vs_sft": (
                float(concentration["effective_task_count_from_hhi"])
                - float(sft_concentration["effective_task_count_from_hhi"])
            ),
            "bootstrap_vs_sft": bootstrap,
            "_raw_summary": raw_summary,
        }
        arms[name] = row
        grouped[scale].append(row)

    scale_summary: dict[str, Any] = {
        "0.0": {
            "scale": 0.0,
            "num_seeds": 1,
            "pass_delta_vs_sft": {
                key: {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
                for key in PASS_KEYS
            },
            "solved_delta_vs_sft": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
            "success_hhi_delta_vs_sft": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
            "effective_task_count_delta_vs_sft": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
            "exact_match_fraction_vs_sft": {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0},
        }
    }

    for scale, rows in sorted(grouped.items()):
        scale_summary[str(scale)] = {
            "scale": scale,
            "num_seeds": len(rows),
            "pass_delta_vs_sft": {
                key: _stats([float(row["pass_delta_vs_sft"][key]) for row in rows])
                for key in PASS_KEYS
            },
            "solved_delta_vs_sft": _stats(
                [float(row["solved_delta_vs_sft"]) for row in rows]
            ),
            "success_hhi_delta_vs_sft": _stats(
                [float(row["success_hhi_delta_vs_sft"]) for row in rows]
            ),
            "effective_task_count_delta_vs_sft": _stats(
                [float(row["effective_task_count_delta_vs_sft"]) for row in rows]
            ),
            "exact_match_fraction_vs_sft": _stats(
                [float(row["exact_match_fraction_vs_sft"]) for row in rows]
            ),
        }

    sft_task_pass = {
        k: _task_passk(sft_summary, k)
        for k in (1, 4, 8, 16)
    }
    for scale, rows in sorted(grouped.items()):
        cluster: dict[str, Any] = {}
        for k in (1, 4, 8, 16):
            ids = sorted(sft_task_pass[k])
            arm_maps = [
                _task_passk(row["_raw_summary"], k)
                for row in rows
            ]
            diffs: list[float] = []
            for task_id in ids:
                seed_deltas = [
                    arm_map[task_id] - sft_task_pass[k][task_id]
                    for arm_map in arm_maps
                ]
                diffs.append(sum(seed_deltas) / len(seed_deltas))
            cluster[f"pass@{k}"] = _cluster_bootstrap(diffs)
        scale_summary[str(scale)]["task_cluster_bootstrap"] = cluster

    for row in arms.values():
        row.pop("_raw_summary", None)

    pass16_curve = [
        {
            "scale": float(info["scale"]),
            "mean_delta": float(info["pass_delta_vs_sft"]["pass@16"]["mean"]),
        }
        for info in scale_summary.values()
    ]
    pass16_curve.sort(key=lambda x: x["scale"])

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
        "by_scale": scale_summary,
        "pass16_curve": pass16_curve,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-eval", type=Path, required=True)
    parser.add_argument("--sft-summary", type=Path, required=True)
    parser.add_argument("--sft-concentration", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = summarize(
        sft_eval=_load(args.sft_eval),
        sft_summary=_load(args.sft_summary),
        sft_concentration=_load(args.sft_concentration),
        analysis_root=args.analysis_root,
        arm_dirs=dict(parse_arm(item) for item in args.arm),
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
