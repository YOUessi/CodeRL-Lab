from __future__ import annotations

import argparse
import json
from pathlib import Path

from .large_k_extend import load_jsonl, verify_prefix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify n=64 continuation exactly preserves the n=16 prefix"
    )
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--extended", type=Path, required=True)
    p.add_argument("--targets", type=Path, required=True)
    p.add_argument("--prefix-samples", type=int, default=16)
    p.add_argument("--output", type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    target_ids = {
        str(row["task_id"]) for row in load_jsonl(args.targets)
    }
    result = verify_prefix(
        original_predictions=load_jsonl(args.original),
        extended_predictions=load_jsonl(args.extended),
        target_ids=target_ids,
        prefix_samples=args.prefix_samples,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if not result["identical"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
