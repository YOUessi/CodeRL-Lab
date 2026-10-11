"""EXP-004P public-only v4 cohort, disjoint from used LCB v5 and v6.

This is an independent older-release comparison, NOT a later temporal benchmark.
Never decode upstream private_test_cases or write raw source into Git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from coderl_lab.datasets.livecodebench import (
    DATASET_REPO, PINNED_LCB_DATASET_REVISION,
    LiveCodeBenchPublicTask, sha256_file,
)

FILES = ("test4.jsonl", "test5.jsonl", "test6.jsonl")


def canonical_key(row: dict) -> str:
    body = " ".join(str(row["question_content"]).split()).casefold()
    starter = " ".join(str(row.get("starter_code", "")).split()).casefold()
    return hashlib.sha256((body + "\n" + starter).encode()).hexdigest()


def public_rows(path: Path) -> list[dict]:
    rows = []
    ids = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            source = json.loads(line)
            # Explicit allow-list: private tests are never decoded or copied.
            public = LiveCodeBenchPublicTask.from_upstream_row(source).to_public_dict()
            if public["question_id"] in ids:
                raise ValueError("duplicate ID in source")
            ids.add(public["question_id"])
            rows.append(public)
    if not rows:
        raise ValueError("empty source")
    return rows


def prepare(sources: dict[str, Path], output: Path) -> dict:
    if set(sources) != set(FILES):
        raise ValueError("requires all three fixed fine-grained releases")
    raw_hashes = {name: sha256_file(path) for name, path in sources.items()}
    public = {name: public_rows(sources[name]) for name in FILES}
    used = public["test5.jsonl"] + public["test6.jsonl"]
    used_ids = {str(r["question_id"]) for r in used}
    used_keys = {canonical_key(r) for r in used}
    accepted = []
    reasons = Counter()
    seen_keys = set()
    for row in public["test4.jsonl"]:
        key = canonical_key(row)
        if str(row["question_id"]) in used_ids:
            reasons["used_question_id"] += 1
        elif key in used_keys:
            reasons["used_normalized_statement"] += 1
        elif key in seen_keys:
            reasons["intra_cohort_duplicate_statement"] += 1
        else:
            accepted.append(row)
            seen_keys.add(key)
    if not accepted:
        raise ValueError("empty independent cohort")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "public_tasks.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "experiment": "EXP-004P",
        "status": "public_only_dataset_preparation",
        "cohort": "LiveCodeBench fine-grained v4 (historical, not newer than v5/v6)",
        "source_revision": PINNED_LCB_DATASET_REVISION,
        "sources_sha256": raw_hashes,
        "source_counts": {name: len(rows) for name, rows in public.items()},
        "accepted_count": len(accepted),
        "exclusions": dict(sorted(reasons.items())),
        "selection": "all v4 except same ID or normalized statement in v5/v6; no outcome filtering",
        "public_sha256": sha256_file(path),
        "platform_counts": dict(Counter(r["platform"] for r in accepted)),
        "difficulty_counts": dict(Counter(r["difficulty"] for r in accepted)),
        "contest_date_min": min(r["contest_date"] for r in accepted),
        "contest_date_max": max(r["contest_date"] for r in accepted),
        "private_tests_decoded": False,
        "private_tests_exported": False,
        "hidden_outcomes_used": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--input-dir", type=Path)
    args = ap.parse_args()
    if args.input_dir is None:
        from huggingface_hub import hf_hub_download
        sources = {
            name: Path(hf_hub_download(
                repo_id=DATASET_REPO,
                filename=name,
                revision=PINNED_LCB_DATASET_REVISION,
                repo_type="dataset",
            )) for name in FILES
        }
    else:
        sources = {name: args.input_dir / name for name in FILES}
    print(json.dumps(prepare(sources, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
