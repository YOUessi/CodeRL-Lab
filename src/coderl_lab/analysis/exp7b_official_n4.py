"""Post-freeze official private LCB scoring for 7B historical v4 n=4.

Never feeds private tests into prompts, generation or parameter selection.
Only aggregate metrics and harmless hashes should be committed to GitHub.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
from collections import Counter
from pathlib import Path

from coderl_lab.analysis.livecodebench_official_eval import (
    OfficialLiveCodeBenchExecutor, decode_test_cases, evaluate_code, load_public_tasks,
)
from coderl_lab.datasets.livecodebench import sha256_file
from coderl_lab.analysis.exp7b_public_smoke import MODEL, REVISION, PUBLIC_SHA

SOURCE_SHA = "d711138ddaebfcf5f8ec6a4283ee677298c0f5c5d374a235af92aaf0584510da"
SOURCE_REV = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
OFFICIAL_SHA = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"


def load_frozen(generation_dir: Path, public_path: Path, lcb_repo: Path) -> tuple[list[dict], dict]:
    if sha256_file(public_path) != PUBLIC_SHA:
        raise ValueError("public dataset SHA drift")
    public = load_public_tasks(public_path)
    if len(public) != 101:
        raise ValueError("full 101 task public view required")
    remote = subprocess.run(
        ["git", "-C", str(lcb_repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if remote != OFFICIAL_SHA:
        raise ValueError("official LCB evaluator revision drift")

    identity = json.loads((generation_dir / "identity.json").read_text())
    wanted = {
        "experiment": "EXP-7B-V4-N4", "model": MODEL, "revision": REVISION,
        "cohort_public_sha256": PUBLIC_SHA, "official_evaluator_sha": OFFICIAL_SHA,
        "samples_per_task": 4, "max_new_tokens": 512,
        "sampler": "temperature=0.8, top_p=0.95, fixed per-task sample seed",
        "prompt": "native Instruct, raw generation unmodified",
        "private_tests_accessed": False,
        "qualification": "historical v4 exploratory; not prospective",
    }
    if identity != wanted:
        raise ValueError("generation protocol identity not frozen/matched")
    checkpoint = generation_dir / "checkpoint.jsonl"
    rows = [json.loads(line) for line in checkpoint.read_text().splitlines() if line.strip()]
    keys = [(x["question_id"], int(x["sample_id"])) for x in rows]
    expected = {(qid, n) for qid in public for n in range(4)}
    if len(rows) != 404 or set(keys) != expected or len(keys) != len(set(keys)):
        raise ValueError("incomplete or duplicated 404 code completions")
    for row in rows:
        if "code" not in row:
            raise ValueError("missing original code completion")
        if hashlib.sha256(row["code"].encode()).hexdigest() != row["completion_sha256"]:
            raise ValueError("original sampled completion mutated")
    frozen = {
        "experiment": "EXP-7B-V4-N4-OFFICIAL",
        "public_view_sha256": PUBLIC_SHA, "upstream_source_sha256": SOURCE_SHA,
        "upstream_source_revision": SOURCE_REV, "official_evaluator_sha": OFFICIAL_SHA,
        "generation_identity_sha256": sha256_file(generation_dir / "identity.json"),
        "generation_checkpoint_sha256": sha256_file(checkpoint),
        "count": 404, "task_count": 101,
        "private_decoded_before_freeze": False,
    }
    return rows, frozen


def bootstrap_at4(tasks: list[dict], iterations: int = 20000, seed: int = 42) -> dict:
    outcomes = [int(x["hidden_solved4"]) for x in tasks]
    rng = random.Random(seed)
    n = len(outcomes)
    values = sorted(sum(outcomes[rng.randrange(n)] for _ in range(n)) / n for _ in range(iterations))
    return {
        "observed": sum(outcomes) / n,
        "ci95_low": values[int(0.025 * iterations)],
        "ci95_high": values[int(0.975 * iterations) - 1],
        "iterations": iterations, "seed": seed,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generation-dir", type=Path, required=True)
    ap.add_argument("--public-tasks", type=Path, required=True)
    ap.add_argument("--source", type=Path)
    ap.add_argument("--lcb-repo", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    rows, frozen = load_frozen(args.generation_dir, args.public_tasks, args.lcb_repo)
    if args.source is None:
        from huggingface_hub import hf_hub_download
        args.source = Path(hf_hub_download(
            repo_id="livecodebench/code_generation_lite",
            filename="test4.jsonl", repo_type="dataset", revision=SOURCE_REV,
        ))
    if sha256_file(args.source) != SOURCE_SHA:
        raise ValueError("source sha changed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    freeze = args.output_dir / "frozen_inputs.json"
    if freeze.exists():
        if json.loads(freeze.read_text()) != frozen:
            raise ValueError("refusing private rescoring from changed generator")
    else:
        freeze.write_text(json.dumps(frozen, indent=2) + "\n")
    # Only after the frozen record exists and all constraints pass is the
    # private_test_cases source field decoded.
    official = OfficialLiveCodeBenchExecutor(
        livecodebench_repo=args.lcb_repo, timeout_seconds=6, memory_limit="1g",
    )
    public = load_public_tasks(args.public_tasks)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["question_id"], []).append(row)

    checkpoint = args.output_dir / "official_checkpoint.jsonl"
    scored: dict[tuple[str, int], dict] = {}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if not line.strip():
                continue
            x = json.loads(line)
            k = (x["question_id"], int(x["sample_id"]))
            if k in scored:
                raise ValueError("duplicate private evaluation record")
            scored[k] = x
    found = set()
    for raw in args.source.open(encoding="utf-8"):
        if not raw.strip():
            continue
        row = json.loads(raw)
        qid = str(row["question_id"])
        if qid not in public:
            continue
        if qid in found:
            raise ValueError("duplicate task in source")
        found.add(qid)
        private = decode_test_cases(row["private_test_cases"])
        for sample in sorted(grouped[qid], key=lambda x: x["sample_id"]):
            key = (qid, int(sample["sample_id"]))
            if key in scored:
                if scored[key]["completion_sha256"] != sample["completion_sha256"]:
                    raise ValueError("private checkpoint completion SHA changed")
                continue
            result = evaluate_code(
                executor=official, task=public[qid], code=sample["code"], private_tests=private,
            )
            error = str(result.results[0]) if result.results else ""
            if error in {"runner-error", "invalid-runner-output"}:
                raise RuntimeError(f"official checker infrastructure failure at {qid}/{key[1]}")
            record = {
                "question_id": qid, "sample_id": key[1],
                "completion_sha256": sample["completion_sha256"],
                "passed": bool(result.passed),
                "resource_limit": error == "candidate-resource-limit",
                "timeout": error == "global-timeout",
            }
            with checkpoint.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            scored[key] = record
        if len(found) % 10 == 0:
            print(f"7B official hidden n4: {len(found)}/101 tasks; scores {len(scored)}/404", flush=True)
    if found != set(public) or len(scored) != 404:
        raise RuntimeError("incomplete frozen official evaluation")

    tasks = []
    for qid in sorted(public):
        samples = [scored[(qid, n)] for n in range(4)]
        generated = grouped[qid]
        tasks.append({
            "question_id": qid,
            "platform": public[qid]["platform"],
            "difficulty": public[qid]["difficulty"],
            "hidden_solved4": any(s["passed"] for s in samples),
            "hidden_correct_samples": sum(s["passed"] for s in samples),
            "public_solved4": any(g["public_pass"] for g in generated),
        })
    total_correct = sum(x["hidden_correct_samples"] for x in tasks)
    result = {
        "experiment": "EXP-7B-V4-N4-OFFICIAL", "status": "completed",
        "data_class": "historical v4 exploratory (not prospective)",
        "model": MODEL, "model_revision": REVISION,
        "n_tasks": 101, "n_samples": 404,
        "official_hidden_sample_correct": total_correct,
        "official_hidden_sample_pass_fraction": total_correct / 404,
        "official_hidden_solved_at4": sum(x["hidden_solved4"] for x in tasks),
        "official_hidden_pass_at4_task_fraction": sum(x["hidden_solved4"] for x in tasks) / 101,
        "public_solved_at4": sum(x["public_solved4"] for x in tasks),
        "bootstrap_hidden_pass_at4": bootstrap_at4(tasks),
        "candidate_resource_limit": sum(x["resource_limit"] for x in scored.values()),
        "candidate_global_timeout": sum(x["timeout"] for x in scored.values()),
        "frozen_inputs": frozen,
        "private_tests_accessed_only_after_freeze": True,
        "limitations": "Existing LCB v4 was used for historical diagnostics; this is not a fresh prospective out-of-distribution study. Instruct prompt differs from older Qwen3 Base experiments.",
        "strata": {
            field: {
                k: {"tasks": len(v), "solved_at4": sum(int(x["hidden_solved4"]) for x in v)}
                for k in sorted({str(x[field]) for x in tasks})
                for v in [[x for x in tasks if str(x[field]) == k]]
            }
            for field in ("difficulty", "platform")
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
