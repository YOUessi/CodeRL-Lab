from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ARMS = (
    "scale025_seed101",
    "scale025_seed202",
    "scale025_seed303",
    "scale050_seed101",
    "scale050_seed202",
    "scale050_seed303",
    "scale100_seed101",
    "scale100_seed202",
    "scale100_seed303",
    "scale200_seed101",
    "scale200_seed202",
    "scale200_seed303",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _index(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        if int(row["sample_id"]) != 0:
            continue
        task_id = str(row["task_id"])
        out[task_id] = row
    return out


def build_susceptibility_labels(
    *,
    greedy_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    reference = _index(greedy_root / "sft.jsonl")
    if not reference:
        raise ValueError("SFT greedy reference is empty")

    arm_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ARMS:
        path = greedy_root / arm / "predictions.jsonl"
        arm_rows[arm] = _index(path)

    per_task: dict[str, Any] = {}
    for task_id, ref in sorted(reference.items()):
        by_arm: dict[str, bool] = {}
        by_scale: dict[str, list[bool]] = {
            "0.25": [],
            "0.5": [],
            "1.0": [],
            "2.0": [],
        }
        ref_raw = str(ref.get("raw_completion", ""))

        for arm in ARMS:
            row = arm_rows[arm].get(task_id)
            if row is None:
                raise ValueError(f"arm {arm} missing task {task_id}")
            changed = str(row.get("raw_completion", "")) != ref_raw
            by_arm[arm] = changed

            if arm.startswith("scale025"):
                by_scale["0.25"].append(changed)
            elif arm.startswith("scale050"):
                by_scale["0.5"].append(changed)
            elif arm.startswith("scale100"):
                by_scale["1.0"].append(changed)
            elif arm.startswith("scale200"):
                by_scale["2.0"].append(changed)

        total_changed = sum(int(x) for x in by_arm.values())
        per_task[task_id] = {
            "task_id": task_id,
            "divergent_arms": total_changed,
            "greedy_divergence_rate": total_changed / len(ARMS),
            "by_scale": {
                scale: {
                    "divergent_arms": sum(int(x) for x in values),
                    "divergence_rate": sum(int(x) for x in values) / len(values),
                }
                for scale, values in by_scale.items()
            },
            "by_arm": by_arm,
        }

    result = {
        "arms": list(ARMS),
        "tasks": len(per_task),
        "per_task": per_task,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "tasks": len(per_task),
                "mean_divergence_rate": (
                    sum(
                        float(row["greedy_divergence_rate"])
                        for row in per_task.values()
                    )
                    / len(per_task)
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--greedy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_susceptibility_labels(
        greedy_root=args.greedy_root,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
