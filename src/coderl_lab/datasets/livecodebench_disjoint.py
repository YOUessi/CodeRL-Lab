"""Frozen public-only disjointness audit for EXP-004N v5 vs used EXP-004L v6.

No raw upstream rows or private testcase payloads are accessed. The v5
evaluation set excludes duplicate question IDs or exact normalized question
content shared with v6 *before any model generation or hidden evaluation*.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from coderl_lab.analysis.livecodebench_official_eval import load_public_tasks
from coderl_lab.datasets.livecodebench import sha256_file


def _normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def _statement_digest(row: dict[str, Any]) -> str:
    body = _normalize(str(row["question_content"]))
    starter = _normalize(str(row.get("starter_code", "")))
    return hashlib.sha256((body + "\n" + starter).encode("utf-8")).hexdigest()


def make_disjoint_audit(
    *,
    v5_rows: list[dict[str, Any]],
    v6_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    def require_unique(rows: list[dict[str, Any]], name: str) -> None:
        ids = [str(x["question_id"]) for x in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate question ID within {name}")
        if not all(str(x.get("question_content", "")).strip() for x in rows):
            raise ValueError(f"missing task statement in {name}")
        if any("private_test_cases" in x for x in rows):
            raise ValueError(f"private data leaked into {name}")

    require_unique(v5_rows, "v5")
    require_unique(v6_rows, "v6")
    if len(v5_rows) != 167 or len(v6_rows) != 175:
        raise ValueError("expected v5 167 / v6 175 full fine-grained slices")

    previous_ids = {str(row["question_id"]) for row in v6_rows}
    previous_statements = {_statement_digest(row) for row in v6_rows}
    kept: list[dict[str, Any]] = []
    excluded = Counter()
    for row in v5_rows:
        if str(row["question_id"]) in previous_ids:
            excluded["same_question_id"] += 1
            continue
        if _statement_digest(row) in previous_statements:
            excluded["same_normalized_problem_text"] += 1
            continue
        kept.append(row)
    if not kept:
        raise ValueError("v5 disjoint evaluation is empty")
    assert set(str(x["question_id"]) for x in kept).isdisjoint(previous_ids)
    assert {_statement_digest(x) for x in kept}.isdisjoint(previous_statements)

    summary = {
        "experiment": "EXP-004N",
        "audit_type": "pre-generation_public_only_cross_release_disjointness",
        "original_v5_tasks": len(v5_rows),
        "comparison_v6_tasks": len(v6_rows),
        "disjoint_v5_tasks": len(kept),
        "excluded": dict(sorted(excluded.items())),
        "excluded_total": sum(excluded.values()),
        "selection_rule": "all pinned v5 fine-grained tasks except ID or exact normalized problem+starter duplication with v6; no difficulty filtering",
        "question_id_intersection_after_filter": 0,
        "normalized_problem_text_intersection_after_filter": 0,
        "private_cases_accessed": False,
        "success_metrics_used_to_select_tasks": False,
    }
    return kept, summary


def write_disjoint_view(
    *,
    v5_path: Path,
    v6_path: Path,
    v5_manifest_path: Path,
    v6_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    v5manifest = json.loads(v5_manifest_path.read_text(encoding="utf-8"))
    v6manifest = json.loads(v6_manifest_path.read_text(encoding="utf-8"))
    pin = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
    for path, manifest, split in (
        (v5_path, v5manifest, "v5"),
        (v6_path, v6manifest, "v6"),
    ):
        if manifest.get("source_revision") != pin:
            raise ValueError(f"{split} source version not pinned")
        if manifest.get("fine_grained_version") != split:
            raise ValueError(f"{split} fine-grained version mismatch")
        if manifest.get("private_tests_exported") is not False:
            raise ValueError(f"{split} exported private tests")
        if manifest.get("private_tests_decoded") is not False:
            raise ValueError(f"{split} decoded private tests")
        if sha256_file(path) != manifest.get("public_view_sha256"):
            raise ValueError(f"{split} public view checksum changed")
    kept, summary = make_disjoint_audit(
        v5_rows=list(load_public_tasks(v5_path).values()),
        v6_rows=list(load_public_tasks(v6_path).values()),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "disjoint_v5_public_tasks.jsonl"
    with out.open("w", encoding="utf-8") as file:
        for row in kept:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary.update({
        "dataset_revision": pin,
        "v5_source_sha256": v5manifest["source_sha256"],
        "v6_source_sha256": v6manifest["source_sha256"],
        "v5_public_sha256": v5manifest["public_view_sha256"],
        "v6_public_sha256": v6manifest["public_view_sha256"],
        "disjoint_v5_public_sha256": sha256_file(out),
    })
    (output_dir / "disjoint_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--v5-public", type=Path, required=True)
    p.add_argument("--v6-public", type=Path, required=True)
    p.add_argument("--v5-manifest", type=Path, required=True)
    p.add_argument("--v6-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    summary = write_disjoint_view(
        v5_path=args.v5_public,
        v6_path=args.v6_public,
        v5_manifest_path=args.v5_manifest,
        v6_manifest_path=args.v6_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
