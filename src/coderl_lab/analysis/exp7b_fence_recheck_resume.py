"""Resume EXP-7B posthoc fence recheck with narrowly classified candidate crashes.

The already-frozen normalizer (exp7b_fence_recheck.py) is NOT changed:
its source SHA is recorded in the existing identity.json. Prior public
checkpoint rows are kept. Only a reproducible candidate-level Python
segfault is treated as a wrong answer; unknown runner errors fail closed.

This is an engineering posthoc recheck of historical v4, not a new holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from coderl_lab.analysis import exp7b_fence_recheck as frozen_v1
from coderl_lab.analysis.exp7b_official_n4 import SOURCE_REV, SOURCE_SHA, bootstrap_at4
from coderl_lab.analysis.livecodebench_official_eval import (
    OfficialLiveCodeBenchExecutor,
    decode_test_cases,
    evaluate_code,
    load_public_tasks,
)
from coderl_lab.datasets.livecodebench import sha256_file


CLASSIFIER_VERSION = "v2_candidate_python_segfault_only"


def attributable_candidate_crash(result) -> bool:
    """A submitted Python program recursing until native interpreter SIGSEGV.

    Docker launch failures, malformed evaluator output, and arbitrary
    infrastructure crashes are NEVER automatically marked candidate wrong.
    """
    if result.passed or tuple(result.results) != ("runner-error",):
        return False
    error = result.metadata.get("error", "")
    return (
        isinstance(error, str)
        and error.startswith("Fatal Python error: Segmentation fault")
        and "Current thread" in error
        and 'File "<string>"' in error
    )


def score_frozen_rows(
    *,
    predictions: list[dict],
    public_path: Path,
    lcb_repo: Path,
    output_dir: Path,
    phase: str,
    source_path: Path | None,
) -> dict:
    public = load_public_tasks(public_path)
    if len(public) != 101:
        raise ValueError("expected frozen historical v4 full set")

    private = {}
    if phase == "private":
        if source_path is None or sha256_file(source_path) != SOURCE_SHA:
            raise ValueError("pinned source hash mismatch")
        # Decoding occurs only after identity, 404 public scores and the
        # pre_private_frozen_inputs.json were verified/persisted by main().
        for line in source_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            source_row = json.loads(line)
            qid = str(source_row["question_id"])
            if qid in private:
                raise ValueError(f"duplicate source question ID {qid}")
            private[qid] = decode_test_cases(source_row["private_test_cases"])
        if set(private) != set(public):
            raise ValueError("private source task set mismatch")

    evaluator = OfficialLiveCodeBenchExecutor(
        livecodebench_repo=lcb_repo, timeout_seconds=6, memory_limit="1g",
    )
    checkpoint = output_dir / (phase + "_checkpoint.jsonl")
    done = frozen_v1.checkpoint_read(checkpoint, predictions)
    expected = {(r["question_id"], r["sample_id"]) for r in predictions}
    if set(done) - expected:
        raise ValueError("unexpected existing checkpoint entries")
    for index, prediction in enumerate(predictions, 1):
        key = (prediction["question_id"], prediction["sample_id"])
        if key in done:
            continue
        checked = evaluate_code(
            executor=evaluator,
            task=public[key[0]],
            code=prediction["code"],
            private_tests=private[key[0]] if phase == "private" else None,
        )
        reason = str(checked.results[0]) if checked.results else ""
        candidate_crash = attributable_candidate_crash(checked)
        if reason in {"runner-error", "invalid-runner-output"} and not candidate_crash:
            # Never silently interpret ambiguous checker failures as wrong
            # answers; does not reveal hidden-case data.
            raise RuntimeError(
                f"unclassified official checker failure for {key}: {reason}"
            )
        record = {
            "question_id": key[0],
            "sample_id": key[1],
            "normalized_sha256": prediction["normalized_sha256"],
            "passed": bool(checked.passed),
            "resource_limited": reason == "candidate-resource-limit",
            "global_timeout": reason == "global-timeout",
            "candidate_python_segfault": candidate_crash,
        }
        with checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        done[key] = record
        if index % 20 == 0:
            print(f"posthoc-{phase}: {len(done)}/{len(predictions)}", flush=True)

    if set(done) != expected:
        raise RuntimeError("incomplete 404-sample checkpoint")

    grouped = {}
    for r in done.values():
        grouped.setdefault(r["question_id"], []).append(r)
    if set(grouped) != set(public):
        raise ValueError("missing scored task")
    tasks = [
        {"question_id": qid, "hidden_solved4": any(x["passed"] for x in group)}
        for qid, group in grouped.items()
    ]
    result = {
        "phase": phase,
        "completed": True,
        "samples": 404,
        "tasks": 101,
        "correct_samples": sum(int(x["passed"]) for x in done.values()),
        "solved_at4": sum(int(x["hidden_solved4"]) for x in tasks),
        "correct_fraction": sum(int(x["passed"]) for x in done.values()) / 404,
        "solved_at4_fraction": sum(int(x["hidden_solved4"]) for x in tasks) / 101,
        "resource_limit_count": sum(int(x.get("resource_limited", False)) for x in done.values()),
        "timeout_count": sum(int(x.get("global_timeout", False)) for x in done.values()),
        "candidate_python_segfault_count": sum(int(x.get("candidate_python_segfault", False)) for x in done.values()),
        "bootstrap_task_ci95": bootstrap_at4(tasks),
        "scored_checkpoint_sha256": sha256_file(checkpoint),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("public", "private"), required=True)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--public-tasks", type=Path, required=True)
    parser.add_argument("--lcb-repo", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    predictions, counts, identity = frozen_v1.freeze(
        args.generation_dir,
        args.public_tasks,
        args.lcb_repo,
        args.output_dir,
    )
    if args.phase == "private":
        public_summary = args.output_dir / "public_summary.json"
        if not public_summary.is_file():
            raise ValueError("full public recheck must finish before private recheck")
        public_result = json.loads(public_summary.read_text(encoding="utf-8"))
        if (
            public_result.get("result", {}).get("samples") != 404
            or public_result.get("result", {}).get("completed") is not True
            or public_result.get("frozen") != identity
            or public_result.get("classifier_version") != CLASSIFIER_VERSION
        ):
            raise ValueError("public results or classifier identity changed")
        marker = args.output_dir / "pre_private_frozen_inputs.json"
        frozen = dict(identity)
        frozen["public_scoring_checkpoint_sha256"] = public_result["result"]["scored_checkpoint_sha256"]
        frozen["checker_classification_version"] = CLASSIFIER_VERSION
        frozen["checker_classifier_source_sha256"] = sha256_file(Path(__file__))
        if marker.exists():
            if json.loads(marker.read_text(encoding="utf-8")) != frozen:
                raise ValueError("pre-private freeze record mismatch")
        else:
            marker.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n")

    source = args.source
    if args.phase == "private" and source is None:
        from huggingface_hub import hf_hub_download
        source = Path(hf_hub_download(
            repo_id="livecodebench/code_generation_lite",
            filename="test4.jsonl", repo_type="dataset", revision=SOURCE_REV,
        ))

    result = score_frozen_rows(
        predictions=predictions,
        public_path=args.public_tasks,
        lcb_repo=args.lcb_repo,
        output_dir=args.output_dir,
        phase=args.phase,
        source_path=source,
    )
    summary = {
        "experiment": "EXP-7B-N4-FENCE-RECHECK",
        "status": "completed",
        "qualification": "POSTHOC format repair, deterministic candidate-crash classification; NOT preregistered benchmark",
        "classifier_version": CLASSIFIER_VERSION,
        "normalization": counts,
        "frozen": identity,
        "result": result,
        "warning": "Markdown normalization performed after original raw-score audit; historical v4 previously seen, not a fresh independent replication.",
    }
    (args.output_dir / f"{args.phase}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "phase": args.phase,
        "scored": result["samples"],
        "correct": result["correct_samples"],
        "solved_at4": result["solved_at4"],
        "candidate_python_segfault_count": result["candidate_python_segfault_count"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
