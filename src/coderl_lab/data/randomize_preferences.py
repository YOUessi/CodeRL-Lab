from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def randomize_labels(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    swap_fraction: float = 0.5,
) -> tuple[list[dict[str, Any]], list[int]]:
    if not 0.0 <= swap_fraction <= 1.0:
        raise ValueError("swap_fraction must be in [0, 1]")

    n = len(rows)
    num_swaps = round(n * swap_fraction)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    swap_set = set(indices[:num_swaps])

    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        new = dict(row)
        new["control_label_swapped"] = i in swap_set
        if i in swap_set:
            new["chosen"], new["rejected"] = (
                str(row["rejected"]),
                str(row["chosen"]),
            )
        out.append(new)
    return out, sorted(swap_set)


def build_control(
    *,
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    seed: int,
    swap_fraction: float,
) -> dict[str, Any]:
    rows = load_jsonl(input_path)
    randomized, swapped = randomize_labels(
        rows,
        seed=seed,
        swap_fraction=swap_fraction,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in randomized:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    same_prompts = all(
        str(a["prompt"]) == str(b["prompt"])
        for a, b in zip(rows, randomized, strict=True)
    )
    same_text_multiset = all(
        sorted([str(a["chosen"]), str(a["rejected"])])
        == sorted([str(b["chosen"]), str(b["rejected"])])
        for a, b in zip(rows, randomized, strict=True)
    )

    summary = {
        "source_pairs": len(rows),
        "seed": seed,
        "swap_fraction_requested": swap_fraction,
        "swapped_pairs": len(swapped),
        "unswapped_pairs": len(rows) - len(swapped),
        "actual_swap_fraction": len(swapped) / len(rows) if rows else 0.0,
        "swapped_indices": swapped,
        "same_prompts": same_prompts,
        "same_chosen_rejected_text_multiset_per_pair": same_text_multiset,
        "source_sha256": sha256(input_path),
        "output_sha256": sha256(output_path),
        "purpose": (
            "Negative control: preserve prompts and candidate texts while "
            "destroying systematic minimal-import repair preference."
        ),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build randomized-label control for EXP-004B"
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--swap-fraction", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = build_control(
        input_path=args.input,
        output_path=args.output,
        summary_path=args.summary,
        seed=args.seed,
        swap_fraction=args.swap_fraction,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
