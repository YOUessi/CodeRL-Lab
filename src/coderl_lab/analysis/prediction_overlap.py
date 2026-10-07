from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def analyze(a_rows: list[dict], b_rows: list[dict]) -> dict:
    a = {
        (str(row["task_id"]), int(row["sample_id"])): row
        for row in a_rows
    }
    b = {
        (str(row["task_id"]), int(row["sample_id"])): row
        for row in b_rows
    }
    keys = sorted(set(a).intersection(b))
    if not keys:
        raise ValueError("no overlapping prediction keys")

    completion_matches = 0
    raw_matches = 0
    changed_tasks: set[str] = set()

    for key in keys:
        completion_same = (
            str(a[key].get("completion", ""))
            == str(b[key].get("completion", ""))
        )
        raw_same = (
            str(a[key].get("raw_completion", ""))
            == str(b[key].get("raw_completion", ""))
        )
        completion_matches += int(completion_same)
        raw_matches += int(raw_same)
        if not completion_same:
            changed_tasks.add(key[0])

    n = len(keys)
    return {
        "rows_compared": n,
        "completion_matches": completion_matches,
        "completion_match_fraction": completion_matches / n,
        "raw_matches": raw_matches,
        "raw_match_fraction": raw_matches / n,
        "changed_rows": n - completion_matches,
        "changed_tasks": len(changed_tasks),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = analyze(read_rows(args.a), read_rows(args.b))
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
