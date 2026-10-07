from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


TEMPS = {
    "t020": 0.2,
    "t050": 0.5,
    "t080": 0.8,
    "t100": 1.0,
}

ARMS = (
    "scale025_seed101",
    "scale050_seed202",
    "scale100_seed202",
    "scale200_seed303",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def index_predictions(path: Path) -> dict[tuple[str, int], str]:
    out: dict[tuple[str, int], str] = {}
    for row in load_jsonl(path):
        key = (str(row["task_id"]), int(row["sample_id"]))
        out[key] = str(row["completion"])
    return out


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def paired_task_bootstrap(
    a: dict[str, float],
    b: dict[str, float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    ids = sorted(set(a) & set(b))
    diffs = [b[x] - a[x] for x in ids]
    observed = mean(diffs)
    rng = random.Random(seed)
    n = len(diffs)
    draws: list[float] = []
    for _ in range(iterations):
        draws.append(mean(diffs[rng.randrange(n)] for _ in range(n)))
    draws.sort()
    return {
        "tasks": n,
        "observed_delta": observed,
        "ci95_low": draws[int(0.025 * iterations)],
        "ci95_high": draws[min(iterations - 1, int(0.975 * iterations))],
        "bootstrap_probability_positive": sum(x > 0 for x in draws) / iterations,
        "iterations": iterations,
        "seed": seed,
    }


def summarize(
    *,
    root: Path,
    greedy_summary_path: Path | None,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "temperatures": {},
        "paired_temperature_contrasts": {},
    }
    task_changed_by_temp: dict[str, dict[str, float]] = {}

    for tag, temperature in TEMPS.items():
        reference = index_predictions(root / tag / "sft" / "predictions.jsonl")
        arm_rows: dict[str, Any] = {}
        per_task_accum: dict[str, list[float]] = {}

        for arm in ARMS:
            candidate = index_predictions(root / tag / arm / "predictions.jsonl")
            keys = sorted(set(reference) & set(candidate))
            if not keys:
                raise ValueError(f"no overlapping predictions for {tag}/{arm}")

            changed = [
                1.0 if reference[key] != candidate[key] else 0.0
                for key in keys
            ]
            by_task: dict[str, list[float]] = {}
            for key, value in zip(keys, changed, strict=True):
                by_task.setdefault(key[0], []).append(value)
                per_task_accum.setdefault(key[0], []).append(value)

            evaluation = load_json(
                root / tag / arm / "evaluation" / "enhanced_summary.json"
            )
            arm_rows[arm] = {
                "exact_match_fraction": 1.0 - mean(changed),
                "changed_fraction": mean(changed),
                "changed_rows": int(sum(changed)),
                "pass_at_1": float(evaluation["pass_at_k"]["pass@1"]),
                "pass_at_4": float(evaluation["pass_at_k"]["pass@4"]),
                "hidden_mean": float(evaluation["hidden_mean_pass_rate"]),
                "solved_tasks": int(evaluation["solved_tasks"]),
                "per_task_changed_fraction": {
                    task_id: mean(values)
                    for task_id, values in sorted(by_task.items())
                },
            }

        task_changed = {
            task_id: mean(values)
            for task_id, values in sorted(per_task_accum.items())
        }
        task_changed_by_temp[tag] = task_changed

        changed_values = [
            float(row["changed_fraction"]) for row in arm_rows.values()
        ]
        p1_values = [float(row["pass_at_1"]) for row in arm_rows.values()]
        p4_values = [float(row["pass_at_4"]) for row in arm_rows.values()]

        result["temperatures"][tag] = {
            "temperature": temperature,
            "arms": arm_rows,
            "aggregate_changed_fraction": stats(changed_values),
            "aggregate_pass_at_1": stats(p1_values),
            "aggregate_pass_at_4": stats(p4_values),
            "task_cluster_changed_fraction_mean": mean(task_changed.values()),
        }

    for a, b in (
        ("t020", "t050"),
        ("t050", "t080"),
        ("t080", "t100"),
        ("t020", "t080"),
        ("t020", "t100"),
    ):
        result["paired_temperature_contrasts"][f"{a}_to_{b}"] = (
            paired_task_bootstrap(
                task_changed_by_temp[a],
                task_changed_by_temp[b],
                iterations=iterations,
                seed=seed,
            )
        )

    if greedy_summary_path is not None and greedy_summary_path.exists():
        greedy = load_json(greedy_summary_path)
        rep_changed = [
            float(greedy["arms"][arm]["greedy_changed_fraction"])
            for arm in ARMS
        ]
        result["greedy_reference"] = {
            "representative_arms": list(ARMS),
            "aggregate_changed_fraction": stats(rep_changed),
        }

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--greedy-summary", type=Path)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = summarize(
        root=args.root,
        greedy_summary_path=args.greedy_summary,
        iterations=args.iterations,
        seed=args.seed,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
