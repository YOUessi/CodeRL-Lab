"""Read-only historical v4 capacity comparison after full frozen official scoring.

EXP-7B is Qwen2.5-7B-Instruct, 4 samples/task and posthoc fence normalization.
EXP-004P is a distinct Qwen3-1.7B-Base SFT, one sample/task, different prompt.
This comparison is EXPLORATORY and does not isolate parameter-count effects.
No private input/output text is exported.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from coderl_lab.datasets.livecodebench import sha256_file


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def index_scores(rows: list[dict], *, label: str) -> dict[tuple[str, int], dict]:
    output = {}
    for row in rows:
        key = (str(row["question_id"]), int(row["sample_id"]))
        if key in output:
            raise ValueError(f"duplicate {label} result {key}")
        output[key] = row
    return output


def paired_bootstrap(values: list[int], iterations: int = 20000, seed: int = 42) -> dict:
    if not values:
        raise ValueError("empty paired bootstrap")
    rng = random.Random(seed)
    n = len(values)
    estimates = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(iterations)
    )
    return {
        "difference_fraction": sum(values) / n,
        "ci95_low": estimates[int(0.025 * iterations)],
        "ci95_high": estimates[int(0.975 * iterations) - 1],
        "iterations": iterations,
        "seed": seed,
    }


def build_report(
    *, public_rows: list[dict], private_rows: list[dict],
    exp004p_rows: list[dict],
) -> dict:
    pub = index_scores(public_rows, label="7B public")
    private = index_scores(private_rows, label="7B private")
    one_point_seven = {str(r["task_id"]): r for r in exp004p_rows}
    if len(one_point_seven) != 101:
        raise ValueError("must use all 101 EXP-004P frozen historical v4 results")
    required = {(qid, sample) for qid in one_point_seven for sample in range(4)}
    if set(pub) != required or set(private) != required:
        raise ValueError("404 public/private samples must have matching IDs")
    if any(
        pub[k]["normalized_sha256"] != private[k]["normalized_sha256"]
        for k in required
    ):
        raise ValueError("public/private scored different code")
    per_task = []
    sample_confusion = Counter()
    for qid in sorted(one_point_seven):
        old = one_point_seven[qid]
        samples = range(4)
        pub_pass = [bool(pub[qid, n]["passed"]) for n in samples]
        hidden_pass = [bool(private[qid, n]["passed"]) for n in samples]
        for a, b in zip(pub_pass, hidden_pass):
            sample_confusion[
                "public_pass" if a else "public_fail",
                "hidden_pass" if b else "hidden_fail",
            ] += 1
        per_task.append({
            "platform": str(old["platform"]),
            "difficulty": str(old["difficulty"]),
            "prior_1p7b_sft512_greedy": bool(old["baseline_correct"]),
            "prior_1p7b_gate_low": bool(old["gated_low_destabilize_correct"]),
            "new_7b_public_solved4": any(pub_pass),
            "new_7b_hidden_solved4": any(hidden_pass),
            "new_7b_public_correct_completions": sum(pub_pass),
            "new_7b_hidden_correct_completions": sum(hidden_pass),
        })
    counts = {
        "old_sft512_single_correct": sum(x["prior_1p7b_sft512_greedy"] for x in per_task),
        "old_gated_low_single_correct": sum(x["prior_1p7b_gate_low"] for x in per_task),
        "7b_public_solved4": sum(x["new_7b_public_solved4"] for x in per_task),
        "7b_hidden_solved4": sum(x["new_7b_hidden_solved4"] for x in per_task),
        "7b_correct_hidden_completions": sum(x["new_7b_hidden_correct_completions"] for x in per_task),
        "7b_public_solved4_but_hidden_unsolved": sum(
            x["new_7b_public_solved4"] and not x["new_7b_hidden_solved4"]
            for x in per_task
        ),
        "7b_new_solved_vs_old_sft_single": sum(
            x["new_7b_hidden_solved4"] and not x["prior_1p7b_sft512_greedy"]
            for x in per_task
        ),
        "old_sft_single_solved_but_7b_no_solved4": sum(
            x["prior_1p7b_sft512_greedy"] and not x["new_7b_hidden_solved4"]
            for x in per_task
        ),
    }
    strata = {}
    for field in ("difficulty", "platform"):
        strata[field] = {}
        for val in sorted({x[field] for x in per_task}):
            subset = [x for x in per_task if x[field] == val]
            strata[field][val] = {
                "tasks": len(subset),
                "old_1p7b_sft_single_correct": sum(x["prior_1p7b_sft512_greedy"] for x in subset),
                "old_1p7b_gated_low_correct": sum(x["prior_1p7b_gate_low"] for x in subset),
                "7b_hidden_solved4": sum(x["new_7b_hidden_solved4"] for x in subset),
                "7b_public_solved4": sum(x["new_7b_public_solved4"] for x in subset),
            }
    differences = [
        int(x["new_7b_hidden_solved4"]) - int(x["prior_1p7b_sft512_greedy"])
        for x in per_task
    ]
    return {
        "experiment": "EXP-7B-V4-POSTHOC-DESCRIPTIVE-CAPACITY-AUDIT",
        "status": "completed",
        "tasks": 101,
        "candidates": 404,
        "comparability_warning": (
            "Historical v4 results already known. 7B Instruct n=4 with posthoc Markdown "
            "normalization compared to different 1.7B Base SFT greedy and gate-low prompts. "
            "Neither pairwise difference nor bootstrap interval identifies parameter-scale "
            "causality, nor constitutes fresh benchmark replication."
        ),
        "counts": counts,
        "sample_level_public_vs_private": {
            f"{a}/{b}": sample_confusion[a, b]
            for a in ("public_pass", "public_fail")
            for b in ("hidden_pass", "hidden_fail")
        },
        "strata": strata,
        "unmatched_model_conditions_paired_descriptive_ci": paired_bootstrap(differences),
        "private_test_payload_exported": False,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--recheck", type=Path, required=True)
    p.add_argument("--prior-details", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    first = json.loads((args.recheck / "public_summary.json").read_text(encoding="utf-8"))
    second = json.loads((args.recheck / "private_summary.json").read_text(encoding="utf-8"))
    if (
        first.get("status") != "completed"
        or second.get("status") != "completed"
        or first.get("frozen") != second.get("frozen")
        or first.get("classifier_version") != second.get("classifier_version")
        or first.get("qualification") != second.get("qualification")
    ):
        raise ValueError("non-matching or unfinished frozen evaluations")
    for part, summary in (("public", first), ("private", second)):
        checkpoint = args.recheck / f"{part}_checkpoint.jsonl"
        if sha256_file(checkpoint) != summary["result"]["scored_checkpoint_sha256"]:
            raise ValueError(f"{part} checkpoint SHA mismatch")
        if summary["result"]["samples"] != 404:
            raise ValueError(f"{part} benchmark incomplete")

    result = build_report(
        public_rows=read_jsonl(args.recheck / "public_checkpoint.jsonl"),
        private_rows=read_jsonl(args.recheck / "private_checkpoint.jsonl"),
        exp004p_rows=read_jsonl(args.prior_details),
    )
    result["source_checkpoints_sha256"] = {
        "7b_public": first["result"]["scored_checkpoint_sha256"],
        "7b_private": second["result"]["scored_checkpoint_sha256"],
        "exp004p_details": sha256_file(args.prior_details),
    }
    if args.output.exists():
        if json.loads(args.output.read_text(encoding="utf-8")) != result:
            raise ValueError("existing immutable analysis output differs")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
