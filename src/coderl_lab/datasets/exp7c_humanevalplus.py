"""Freeze prompt-only HumanEval+ cohort. Never export solutions or tests.

The pinned upstream source contains official benchmark answers/test inputs.
Only allowlisted task IDs, function signatures and prompts enter generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

REPO = "evalplus/humanevalplus"
REVISION = "aa0d916268b1c17e84e881e9bd460508dd2fd308"
FILE = "test.jsonl"
SOURCE_SHA256 = "908377f1daf28dcb36846db73a5662b2e05a9907407c2696c89ad9d3b0b04492"
ALLOWED_FIELDS = frozenset(("task_id", "prompt", "entry_point"))
FORBIDDEN_FIELDS = frozenset((
    "canonical_solution", "test", "base_input", "plus_input",
    "reference_solution", "solution", "expected_output",
))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def public_rows_from_source(
    source: list[dict[str, Any]], *, expected_count: int = 164
) -> list[dict[str, str]]:
    if len(source) != expected_count:
        raise ValueError(f"expected {expected_count} source tasks, got {len(source)}")
    entries: dict[str, dict[str, str]] = {}
    for raw in source:
        if not isinstance(raw, dict):
            raise ValueError("source must contain JSON objects only")
        missing = ALLOWED_FIELDS - raw.keys()
        if missing:
            raise ValueError(f"source task missing {sorted(missing)}")
        task_id = str(raw["task_id"])
        prompt = str(raw["prompt"])
        entry_point = str(raw["entry_point"])
        if not re.fullmatch(r"HumanEval/(0|[1-9][0-9]*)", task_id):
            raise ValueError(f"non-HumanEval task ID: {task_id}")
        if task_id in entries:
            raise ValueError(f"duplicated task ID: {task_id}")
        if not prompt.strip() or not entry_point.isidentifier():
            raise ValueError(f"invalid prompt/function name: {task_id}")
        if not re.search(r"\\bdef\\s+" + re.escape(entry_point) + r"\\s*\\(", prompt):
            raise ValueError(f"entry point signature not in original prompt: {task_id}")
        entries[task_id] = {
            "task_id": task_id,
            "prompt": prompt,
            "entry_point": entry_point,
        }
    return [entries[qid] for qid in sorted(
        entries, key=lambda x: int(x.split("/")[1])
    )]


def write_public(rows: list[dict[str, str]], out_dir: Path) -> dict[str, Any]:
    if any(set(r) != ALLOWED_FIELDS for r in rows):
        raise ValueError("public whitelist violated")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "public_tasks.jsonl"
    if path.exists():
        raise FileExistsError("public cohort already frozen: refusing replacement")
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(
                row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ) + "\n")
    manifest = {
        "experiment": "EXP-7C",
        "benchmark": "HumanEval+",
        "data_source": REPO,
        "source_revision": REVISION,
        "source_file": FILE,
        "source_sha256": SOURCE_SHA256,
        "tasks": len(rows),
        "public_keys": sorted(ALLOWED_FIELDS),
        "public_sha256": sha256_file(path),
        "reference_solutions_exported": False,
        "base_inputs_exported": False,
        "plus_inputs_exported": False,
        "official_tests_accessed_during_generation": False,
        "benchmark_claim": "new to CodeRL-Lab; historical public benchmark, not forward temporal",
    }
    (out_dir / "manifest.json").write_text(json.dumps(
        manifest, ensure_ascii=False, indent=2
    ) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--source-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args=p.parse_args()
    source_sha=sha256_file(args.source_jsonl)
    if source_sha != SOURCE_SHA256:
        raise ValueError(f"HumanEval+ pinned source SHA mismatch: {source_sha}")
    source = [
        json.loads(line)
        for line in args.source_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    public=public_rows_from_source(source)
    manifest=write_public(public, args.output_dir)
    print(json.dumps(manifest, indent=2),flush=True)


if __name__=="__main__":
    main()
