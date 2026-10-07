from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize_completion(text: str) -> str:
    lines = [line.rstrip() for line in str(text).strip().splitlines()]
    return "\n".join(lines)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_id"])].append(
            normalize_completion(str(row["completion"]))
        )
    if not grouped:
        raise ValueError("no predictions")

    per_task: dict[str, Any] = {}
    for task_id, comps in sorted(grouped.items()):
        counts = Counter(comps)
        n = len(comps)
        probs = [count / n for count in counts.values()]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        max_entropy = math.log(n) if n > 1 else 0.0
        modal_fraction = max(counts.values()) / n
        unique = len(counts)
        per_task[task_id] = {
            "samples": n,
            "unique_completions": unique,
            "unique_fraction": unique / n,
            "empirical_entropy_nats": entropy,
            "normalized_entropy": entropy / max_entropy if max_entropy > 0 else 0.0,
            "modal_fraction": modal_fraction,
            "duplicate_fraction": 1.0 - unique / n,
        }

    keys = [
        "unique_completions",
        "unique_fraction",
        "empirical_entropy_nats",
        "normalized_entropy",
        "modal_fraction",
        "duplicate_fraction",
    ]
    return {
        "tasks": len(per_task),
        "mean": {
            key: mean(float(row[key]) for row in per_task.values())
            for key in keys
        },
        "per_task": per_task,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize exact-completion diversity")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = summarize(load_jsonl(args.predictions))
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
