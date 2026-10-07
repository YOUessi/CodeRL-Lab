from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from safetensors import safe_open


def parse_arm(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("arm must be NAME=PATH")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("arm name cannot be empty")
    return name, Path(path)


def analyze_delta_directions(
    *,
    base_path: Path,
    arms: dict[str, Path],
) -> dict[str, Any]:
    if len(arms) < 2:
        raise ValueError("at least two arms are required")

    names = list(arms)
    norm2 = {name: 0.0 for name in names}
    dot = {
        (a, b): 0.0
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    }

    with safe_open(base_path, framework="pt", device="cpu") as base_file:
        arm_files = {
            name: safe_open(path, framework="pt", device="cpu")
            for name, path in arms.items()
        }
        try:
            base_keys = list(base_file.keys())
            for name, arm_file in arm_files.items():
                if list(arm_file.keys()) != base_keys:
                    raise ValueError(
                        f"tensor keys differ between base and arm {name}"
                    )

            for key in base_keys:
                base_tensor = base_file.get_tensor(key).float()
                deltas = {}
                for name, arm_file in arm_files.items():
                    delta = (
                        arm_file.get_tensor(key).float() - base_tensor
                    ).flatten()
                    deltas[name] = delta
                    norm2[name] += float((delta * delta).sum().item())

                for i, a in enumerate(names):
                    for b in names[i + 1 :]:
                        dot[(a, b)] += float(
                            (deltas[a] * deltas[b]).sum().item()
                        )
        finally:
            for arm_file in arm_files.values():
                arm_file.__exit__(None, None, None)

    cosine: dict[str, float | None] = {}
    for (a, b), value in dot.items():
        denominator = math.sqrt(norm2[a] * norm2[b])
        cosine[f"{a}__{b}"] = (
            value / denominator if denominator > 0 else None
        )

    return {
        "base_path": str(base_path),
        "delta_l2": {
            name: math.sqrt(value) for name, value in norm2.items()
        },
        "pairwise_cosine": cosine,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare LoRA adapter update directions relative to a base"
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        help="Repeated NAME=PATH adapter specification.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arms = dict(parse_arm(value) for value in args.arm)
    result = analyze_delta_directions(
        base_path=args.base,
        arms=arms,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
