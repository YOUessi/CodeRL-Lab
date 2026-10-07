from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .completion_diversity import load_jsonl, summarize


def _index(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["task_id"]), int(row["sample_id"]))
        if key in out:
            raise ValueError(f"duplicate prediction key: {key}")
        out[key] = row
    return out


def compare_behavior(
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    ref = _index(reference_rows)
    cand = _index(candidate_rows)

    if set(ref) != set(cand):
        missing = sorted(set(ref) - set(cand))
        extra = sorted(set(cand) - set(ref))
        raise ValueError(
            f"prediction key mismatch: missing={missing[:10]}, extra={extra[:10]}"
        )

    keys = sorted(ref)
    exact = [
        key for key in keys
        if str(ref[key]["completion"]) == str(cand[key]["completion"])
    ]
    raw_exact = [
        key for key in keys
        if str(ref[key].get("raw_completion", ""))
        == str(cand[key].get("raw_completion", ""))
    ]
    changed = [key for key in keys if key not in set(exact)]
    changed_tasks = sorted({task_id for task_id, _ in changed})

    ref_div = summarize(reference_rows)
    cand_div = summarize(candidate_rows)

    diversity_delta = {
        key: (
            float(cand_div["mean"][key])
            - float(ref_div["mean"][key])
        )
        for key in ref_div["mean"]
    }

    return {
        "rows_compared": len(keys),
        "exact_completion_matches": len(exact),
        "exact_completion_match_fraction": len(exact) / len(keys),
        "exact_raw_matches": len(raw_exact),
        "exact_raw_match_fraction": len(raw_exact) / len(keys),
        "changed_rows": len(changed),
        "changed_tasks": len(changed_tasks),
        "changed_task_ids": changed_tasks,
        "reference_diversity": ref_div["mean"],
        "candidate_diversity": cand_div["mean"],
        "candidate_minus_reference_diversity": diversity_delta,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare fixed-seed generation behavior between two policies"
    )
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = compare_behavior(
        load_jsonl(args.reference),
        load_jsonl(args.candidate),
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
