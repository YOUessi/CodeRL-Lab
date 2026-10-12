"""EXP-7B posthoc evaluation-contract repair.

Retrospective engineering audit only: same frozen 404 completions; remove a
single complete leading Markdown code block without changing executable content.
Never treat these labels as the preregistered accuracy or a fresh holdout.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

from coderl_lab.analysis.exp7b_official_n4 import (
    OFFICIAL_SHA, SOURCE_REV, SOURCE_SHA, bootstrap_at4, load_frozen,
)
from coderl_lab.analysis.livecodebench_official_eval import (
    OfficialLiveCodeBenchExecutor, decode_test_cases, evaluate_code, load_public_tasks,
)
from coderl_lab.datasets.livecodebench import sha256_file

FENCE = chr(96) * 3
OPEN = re.compile(r"^" + re.escape(FENCE) + r"(?:python|py)?[ \t]*\r?\n", re.I)
CLOSE = re.compile(r"\r?\n" + re.escape(FENCE) + r"(?=[ \t]*(?:\r?\n|$))")


def code_without_outer_fence(raw: str) -> tuple[str, str]:
    """Deterministically select the first fully closed opening code fence."""
    stripped = raw.strip()
    opening = OPEN.match(stripped)
    if opening is None:
        return raw, "no_leading_python_fence"
    closing = CLOSE.search(stripped, opening.end())
    if closing is None:
        return raw, "unclosed_fence_unchanged"
    # Ignore text outside the first COMPLETE code fence, never edit the code body.
    return stripped[opening.end():closing.start()].strip(), "first_closed_python_fence"


def can_parse_python(code: str) -> bool:
    try:
        ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    return True


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def format_predictions(original: list[dict]) -> tuple[list[dict], dict]:
    counts: Counter[str] = Counter()
    transformed = []
    for row in original:
        raw = row["code"]
        code, reason = code_without_outer_fence(raw)
        counts[reason] += 1
        counts["original_python_ast_valid"] += int(can_parse_python(raw))
        counts["normalized_python_ast_valid"] += int(can_parse_python(code))
        counts["output_changed"] += int(code != raw)
        counts["reached_512_token_cap"] += int(int(row["tokens"]) >= 512)
        transformed.append({
            "question_id": row["question_id"],
            "sample_id": int(row["sample_id"]),
            "original_sha256": row["completion_sha256"],
            "normalized_sha256": digest(code),
            "code": code,
            "format_transform": reason,
            "public_original_pass": bool(row["public_pass"]),
        })
    return transformed, dict(counts)


def freeze(
    generation_dir: Path, public_path: Path, lcb_repo: Path, out: Path
) -> tuple[list[dict], dict, dict]:
    original, parent = load_frozen(generation_dir, public_path, lcb_repo)
    normalized, counts = format_predictions(original)
    out.mkdir(parents=True, exist_ok=True)
    identity = {
        "experiment": "EXP-7B-N4-FENCE-RECHECK",
        "protocol_type": "posthoc_evaluation_format_repair_not_prospective",
        "source_checkpoint_sha256": parent["generation_checkpoint_sha256"],
        "source_identity_sha256": parent["generation_identity_sha256"],
        "normalizer_module_sha256": sha256_file(Path(__file__)),
        "cohort_public_sha256": parent["public_view_sha256"],
        "official_evaluator_sha256": OFFICIAL_SHA,
        "candidate_count": 404,
        "transform": "select exactly the first complete leading python fenced block; otherwise preserve raw code",
        "model_generation_repeated": False,
        "selection_based_on_hidden_labels": False,
    }
    p = out / "identity.json"
    if p.exists():
        if json.loads(p.read_text()) != identity:
            raise ValueError("posthoc recheck identity changed")
    else:
        p.write_text(json.dumps(identity, indent=2) + "\n")
    q = out / "normalized_predictions.jsonl"
    expected = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in normalized)
    if q.exists():
        if q.read_text() != expected:
            raise ValueError("normalized model completions changed")
    else:
        q.write_text(expected)
    identity["normalized_predictions_sha256"] = sha256_file(q)
    return normalized, counts, identity


def checkpoint_read(path: Path, expected: list[dict]) -> dict[tuple[str, int], dict]:
    entries = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if not line:
                continue
            row = json.loads(line)
            key = (str(row["question_id"]), int(row["sample_id"]))
            if key in entries:
                raise ValueError("duplicate checkpoint key")
            entries[key] = row
    allowed = {(x["question_id"], x["sample_id"]): x for x in expected}
    for key, row in entries.items():
        if key not in allowed or row["normalized_sha256"] != allowed[key]["normalized_sha256"]:
            raise ValueError("checkpoint code does not match frozen normalized generation")
    return entries


def score(
    *, normalized: list[dict], public_path: Path, lcb_repo: Path,
    out: Path, phase: str, source_path: Path | None,
) -> dict:
    public = load_public_tasks(public_path)
    private = {}
    if phase == "private":
        # This is called after freeze, public scoring, and source-SHA checks.
        if source_path is None or sha256_file(source_path) != SOURCE_SHA:
            raise ValueError("pinned test4 source missing or mismatch")
        for line in source_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["question_id"])
            if qid in private:
                raise ValueError("repeated task in official source")
            private[qid] = decode_test_cases(row["private_test_cases"])
        if set(private) != set(public):
            raise ValueError("private source does not match 101 frozen tasks")

    judge = OfficialLiveCodeBenchExecutor(
        livecodebench_repo=lcb_repo, timeout_seconds=6, memory_limit="1g"
    )
    checkpoint = out / (phase + "_checkpoint.jsonl")
    done = checkpoint_read(checkpoint, normalized)
    total = len(normalized)
    for i, row in enumerate(normalized):
        key = (row["question_id"], row["sample_id"])
        if key in done:
            continue
        check = evaluate_code(
            executor=judge, task=public[row["question_id"]],
            code=row["code"],
            private_tests=private[row["question_id"]] if phase == "private" else None,
        )
        reason = str(check.results[0]) if check.results else ""
        if reason in {"runner-error", "invalid-runner-output"}:
            raise RuntimeError(f"official checker failure for {key}: {reason}")
        record = {
            "question_id": key[0], "sample_id": key[1],
            "normalized_sha256": row["normalized_sha256"],
            "passed": bool(check.passed),
            "resource_limited": reason == "candidate-resource-limit",
            "global_timeout": reason == "global-timeout",
        }
        with checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        done[key] = record
        if (i+1) % 20 == 0:
            print(f"posthoc-{phase}: {i+1}/{total}", flush=True)
    if len(done) != 404:
        raise RuntimeError("posthoc check incomplete")
    grouped: dict[str, list[dict]] = {}
    for item in done.values():
        grouped.setdefault(item["question_id"], []).append(item)
    task_rows = [{"question_id":qid,"hidden_solved4":any(x["passed"] for x in samples)}
                 for qid,samples in grouped.items()]
    result = {
        "phase": phase, "completed": True, "samples": 404, "tasks": 101,
        "correct_samples": sum(int(x["passed"]) for x in done.values()),
        "solved_at4": sum(int(x["hidden_solved4"]) for x in task_rows),
        "correct_fraction": sum(int(x["passed"]) for x in done.values()) / 404,
        "solved_at4_fraction": sum(int(x["hidden_solved4"]) for x in task_rows) / 101,
        "resource_limit_count": sum(int(x["resource_limited"]) for x in done.values()),
        "timeout_count": sum(int(x["global_timeout"]) for x in done.values()),
        "bootstrap_task_ci95": bootstrap_at4(task_rows),
        "scored_checkpoint_sha256": sha256_file(checkpoint),
    }
    return result


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--phase", choices=["public","private"], required=True)
    p.add_argument("--generation-dir", type=Path, required=True)
    p.add_argument("--public-tasks", type=Path, required=True)
    p.add_argument("--lcb-repo", type=Path, required=True)
    p.add_argument("--source", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    args=p.parse_args()

    normalized, counts, identity = freeze(
        args.generation_dir, args.public_tasks, args.lcb_repo, args.output_dir
    )
    if args.phase == "private":
        if not (args.output_dir / "public_summary.json").is_file():
            raise ValueError("public normalized scoring not completed before private")
        pub = json.loads((args.output_dir / "public_summary.json").read_text())
        if pub.get("samples") != 404 or pub.get("completed") is not True:
            raise ValueError("public normalized scoring incomplete")
        # Preserve a freeze barrier before decoding any private test cases.
        marker = args.output_dir / "pre_private_frozen_inputs.json"
        frozen = dict(identity)
        frozen["public_scoring_checkpoint_sha256"] = pub["scored_checkpoint_sha256"]
        if marker.exists():
            if json.loads(marker.read_text()) != frozen:
                raise ValueError("pre-private frozen inputs changed")
        else:
            marker.write_text(json.dumps(frozen, indent=2) + "\n")
    source=args.source
    if args.phase == "private" and source is None:
        from huggingface_hub import hf_hub_download
        source=Path(hf_hub_download(
            repo_id="livecodebench/code_generation_lite",
            filename="test4.jsonl", repo_type="dataset", revision=SOURCE_REV
        ))
    result = score(
        normalized=normalized, public_path=args.public_tasks,
        lcb_repo=args.lcb_repo, out=args.output_dir,
        phase=args.phase, source_path=source
    )
    summary = {
        "experiment": "EXP-7B-N4-FENCE-RECHECK", "status":"completed",
        "qualification": "POSTHOC EVALUATOR INPUT NORMALIZATION; not the preregistered result",
        "normalization": counts, "frozen": identity,
        "result": result,
        "warning": "All 404 raw completions contained markdown fences. Do not compare this posthoc adjusted score as independently confirmed improvement.",
    }
    path=args.output_dir / (args.phase + "_summary.json")
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "phase":args.phase, "normalization":counts,
        "correct_samples":result["correct_samples"],
        "solved_at4":result["solved_at4"],
    },ensure_ascii=False,indent=2), flush=True)


if __name__=="__main__":
    main()
