"""Post-freeze official private evaluation of EXP-004N preregistered v5 arms.

No private test payload is ever persisted. CPU summary code is directly testable
with synthetic examples. The source/manifest/runner SHA audit runs before
decoding even one private test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from coderl_lab.analysis.bottleneck_correctness_causality import bootstrap_task_cluster
from coderl_lab.analysis.livecodebench_official_eval import (
    OfficialLiveCodeBenchExecutor, decode_test_cases, evaluate_code, load_public_tasks,
)
from coderl_lab.datasets.livecodebench import sha256_file
from coderl_lab.analysis.exp004n_capacity_generate import (
    SOURCE_SHA256, PUBLIC_SHA256, SOURCE_REVISION,
    OFFICIAL_COMMIT, MODEL, MODEL_REVISION, SFT_ADAPTER_SHA256,
)
from coderl_lab.analysis.livecodebench_formal_eval import transitions

ARMS = ("base512", "sft512", "gated_low", "gated_high", "always_low", "sft1024")
SOURCE_FILENAME = "test5.jsonl"


def sha_string(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_inputs(
    *,
    sft512_path: Path, base512_path: Path, sft1024_path: Path,
    public_path: Path, manifest_path: Path,
    adapter_dir: Path, lcb_repo: Path,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if any((
        manifest.get("fine_grained_version") != "v5",
        manifest.get("source_filename") != SOURCE_FILENAME,
        manifest.get("source_revision") != SOURCE_REVISION,
        manifest.get("source_sha256") != SOURCE_SHA256,
        manifest.get("public_view_sha256") != PUBLIC_SHA256,
        manifest.get("tasks") != 167,
        manifest.get("private_tests_exported") is not False,
        manifest.get("private_tests_decoded") is not False,
    )):
        raise ValueError("frozen independent v5 dataset does not match preregistration")
    if sha256_file(public_path) != PUBLIC_SHA256:
        raise ValueError("v5 public view changed since preregistration")
    if sha256_file(adapter_dir / "adapter_model.safetensors") != SFT_ADAPTER_SHA256:
        raise ValueError("SFT adapter changed")
    git = subprocess.run(["git", "-C", str(lcb_repo), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True)
    if git.stdout.strip() != OFFICIAL_COMMIT:
        raise ValueError("official checker changed")
    pub = load_public_tasks(public_path)
    if len(pub) != 167:
        raise ValueError("public task set must be 167")
    sft = read_json(sft512_path)
    base = read_json(base512_path)
    long = read_json(sft1024_path)
    for runner in (sft, base, long):
        if runner.get("experiment") != "EXP-004N":
            raise ValueError("wrong experiment source")
        if runner.get("private_tests_accessed") is not False:
            raise ValueError("private tests accessed during generation")
        if runner.get("tasks") != 167 or set(runner["per_task"]) != set(pub):
            raise ValueError("incomplete/mismatched v5 task set")
        if runner.get("revision", runner.get("model_revision")) != MODEL_REVISION:
            raise ValueError("pinned model revision changed")
        if runner.get("official_livecodebench_commit", runner.get("official_checker_commit")) != OFFICIAL_COMMIT:
            raise ValueError("pinned official evaluator changed")
    if any((
        sft.get("max_new_tokens") != 512,
        sft.get("primary_window") != 128,
        sft.get("low_threshold") != 0.05,
        sft.get("high_threshold") != 0.20,
        sft.get("bias") != 0.25,
        sft.get("model") != MODEL,
        base.get("arm") != "base512",
        long.get("arm") != "sft1024",
        base.get("max_new_tokens") != 512,
        long.get("max_new_tokens") != 1024,
        not base.get("is_formal"),
        not long.get("is_formal"),
        base.get("adapter_sha256") is not None,
        long.get("adapter_sha256") != SFT_ADAPTER_SHA256,
        base.get("public_view_sha256") != PUBLIC_SHA256,
        long.get("public_view_sha256") != PUBLIC_SHA256,
    )):
        raise ValueError("frozen model/arm/decoding rule changed")
    for qid in pub:
        task = sft["per_task"][qid]
        if bool(task["gate_triggered"]) != (
            not bool(task["public_all_pass"]) and bool(task["eligible"])
        ):
            raise ValueError("public-only gate mismatch")
        if not task["gate_triggered"]:
            for cond in ("gated_low_destabilize", "gated_high_destabilize"):
                if task[cond]["raw_completion"] != task["baseline"]["raw_completion"]:
                    raise ValueError("public Gate rewrote protected baseline")
        for runner in (base, long):
            if runner["per_task"][qid]["question_id"] != qid:
                raise ValueError("cross-arm task mismatch")
        if bool(base["per_task"][qid]["reached_token_cap"]) != (
            base["per_task"][qid]["generated_token_count"] >= 512
        ):
            raise ValueError("base token cap reporting mismatch")
        if bool(long["per_task"][qid]["reached_token_cap"]) != (
            long["per_task"][qid]["generated_token_count"] >= 1024
        ):
            raise ValueError("long token cap reporting mismatch")
    return {
        "experiment": "EXP-004N",
        "stage": "frozen-before-private-tests",
        "tasks": 167,
        "source_sha256": SOURCE_SHA256,
        "public_view_sha256": PUBLIC_SHA256,
        "official_commit": OFFICIAL_COMMIT,
        "adapter_sha256": SFT_ADAPTER_SHA256,
        "sft512_runner_sha256": sha256_file(sft512_path),
        "base512_runner_sha256": sha256_file(base512_path),
        "sft1024_runner_sha256": sha256_file(sft1024_path),
        "private_tests_accessed": False,
    }


def code_map(qid: str, sft: dict[str, Any], base: dict[str, Any], long: dict[str, Any]) -> dict[str, str]:
    row = sft["per_task"][qid]
    return {
        "base512": str(base["per_task"][qid]["code"]),
        "sft512": str(row["baseline"]["code"]),
        "gated_low": str(row["gated_low_destabilize"]["code"]),
        "gated_high": str(row["gated_high_destabilize"]["code"]),
        "always_low": str(row["always_low_destabilize"]["code"]),
        "sft1024": str(long["per_task"][qid]["code"]),
    }


def summarize(rows: list[dict[str, Any]], *, seed: int = 42, iterations: int = 20000) -> dict[str, Any]:
    if not rows:
        raise ValueError("no completed tasks")
    count = len(rows)
    result: dict[str, Any] = {
        "experiment": "EXP-004N",
        "tasks": count,
        "arms": {}, "contrasts": {}, "strata": {},
    }
    for arm in ARMS:
        correct = sum(bool(row[f"{arm}_correct"]) for row in rows)
        result["arms"][arm] = {
            "correct": correct, "accuracy": correct / count,
            "transitions_vs_sft512": (
                None if arm == "sft512" else transitions(
                    [dict(
                        baseline_correct=r["sft512_correct"],
                        **{f"{arm}_correct":r[f"{arm}_correct"]},
                    ) for r in rows], arm
                )
            ),
        }
    for a, b in (
        ("gated_low", "sft512"), ("gated_high", "sft512"),
        ("always_low", "sft512"), ("base512", "sft512"),
        ("sft1024", "sft512"), ("gated_low", "gated_high"),
        ("gated_low", "always_low"),
    ):
        result["contrasts"][f"{a}_minus_{b}"] = bootstrap_task_cluster(
            rows,
            value_fn=lambda row, lhs=a, rhs=b:
            int(row[f"{lhs}_correct"]) - int(row[f"{rhs}_correct"]),
            iterations=iterations, seed=seed,
        )
    result["preregistered_primary_success"] = (
        result["contrasts"]["gated_low_minus_sft512"]["ci95_low"] > 0
    )
    for field in ("platform", "difficulty"):
        for value in sorted({str(row[field]) for row in rows}):
            chosen = [r for r in rows if str(r[field]) == value]
            result["strata"][f"{field}:{value}"] = {
                "tasks": len(chosen),
                **{arm: sum(bool(x[f"{arm}_correct"]) for x in chosen) for arm in ARMS},
            }
    result["gate_audit"] = {
        "gate_triggered": sum(bool(x["gate_triggered"]) for x in rows),
        "gate_changed": sum(bool(x["gated_low_changed"]) for x in rows),
        "gate_selected_wrong": sum(bool(x["gate_triggered"]) and not x["sft512_correct"] for x in rows),
        "public_pass_protected": all(
            not x["gated_low_changed"] and not x["gated_high_changed"]
            for x in rows if x["sft512_public_all_pass"]
        ),
    }
    result["completion_quality"] = {
        arm: {
            "cap_hit": sum(bool(x[f"{arm}_cap_hit"]) for x in rows),
            "ast_syntax_ok": sum(bool(x[f"{arm}_syntax_ok"]) for x in rows),
        }
        for arm in ("base512", "sft1024")
    }
    result["sft_1024_preserves_512_raw_prefix_count"] = sum(
        bool(x["sft1024_prefix_matches_sft512"]) for x in rows
    )
    result["bootstrap"] = {"seed": seed, "iterations": iterations}
    result["conclusion_limit"] = (
        "independent earlier v5 diagnostic; not a forward-time no-contamination benchmark"
    )
    return result


def run_final(
    *,
    sft512_path: Path, base512_path: Path, sft1024_path: Path,
    public_path: Path, manifest_path: Path, adapter_dir: Path, lcb_repo: Path,
    source_path: Path, output_dir: Path, seed: int = 42, iterations: int = 20000,
) -> dict[str, Any]:
    frozen = validate_inputs(
        sft512_path=sft512_path, base512_path=base512_path,
        sft1024_path=sft1024_path, public_path=public_path,
        manifest_path=manifest_path, adapter_dir=adapter_dir, lcb_repo=lcb_repo,
    )
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("source mutated before private evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = output_dir / "frozen_inputs.json"
    if freeze_path.exists():
        if read_json(freeze_path) != frozen:
            raise ValueError("frozen inputs changed after private outcome checks")
    else:
        freeze_path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    # NEVER decode private tests above this barrier.
    sft = read_json(sft512_path)
    base = read_json(base512_path)
    long = read_json(sft1024_path)
    public = load_public_tasks(public_path)
    checkpoint = output_dir / "checkpoint.jsonl"
    cache: dict[tuple[str, str], bool] = {}
    failed_reason: dict[tuple[str, str], str] = {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = (str(record["qid"]), str(record["code_sha256"]))
            passed = bool(record["passed"])
            if key in cache and cache[key] != passed:
                raise ValueError("contradictory private outcome cache")
            cache[key] = passed
            failed_reason[key] = str(record.get("reason", "none"))
    evaluator = OfficialLiveCodeBenchExecutor(
        livecodebench_repo=lcb_repo, timeout_seconds=6, memory_limit="1g",
    )
    seen: set[str] = set()
    for line in source_path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        qid = str(row["question_id"])
        if qid not in public:
            continue
        if qid in seen:
            raise ValueError("duplicate official question ID")
        seen.add(qid)
        # Phase frozen; safe to FIRST decode original source private field now.
        private = decode_test_cases(row["private_test_cases"])
        codes = code_map(qid, sft, base, long)
        for arm in ARMS:
            code = codes[arm]
            key = (qid, sha_string(code))
            if key in cache:
                continue
            outcome = evaluate_code(
                executor=evaluator, task=public[qid],
                code=code, private_tests=private,
            )
            status = str(outcome.results[0]) if outcome.results else ""
            if status in {"runner-error", "invalid-runner-output", "global-timeout"}:
                raise RuntimeError(f"official evaluator infrastructure error at {qid}/{arm}")
            reason = "candidate-resource-limit" if status == "candidate-resource-limit" else "none"
            record = {
                "qid": qid, "code_sha256": key[1],
                "passed": bool(outcome.passed), "reason": reason,
            }
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            cache[key] = bool(outcome.passed)
            failed_reason[key] = reason
        if len(seen) % 10 == 0:
            print(f"EXP-004N official private: {len(seen)}/167", flush=True)
    if seen != set(public):
        raise ValueError("official source missing preregistered tasks")

    details: list[dict[str, Any]] = []
    for qid in sorted(public):
        row = sft["per_task"][qid]
        b = base["per_task"][qid]
        l = long["per_task"][qid]
        codes = code_map(qid, sft, base, long)
        x: dict[str, Any] = {
            "task_id": qid,
            "qid": qid,
            "platform": row["platform"],
            "difficulty": row["difficulty"],
            "gate_triggered": bool(row["gate_triggered"]),
            "sft512_public_all_pass": bool(row["public_all_pass"]),
            "gated_low_changed": row["gated_low_destabilize"]["raw_completion"] != row["baseline"]["raw_completion"],
            "gated_high_changed": row["gated_high_destabilize"]["raw_completion"] != row["baseline"]["raw_completion"],
            "base512_cap_hit": bool(b["reached_token_cap"]),
            "sft1024_cap_hit": bool(l["reached_token_cap"]),
            "base512_syntax_ok": bool(b["syntax_ok"]),
            "sft1024_syntax_ok": bool(l["syntax_ok"]),
            "sft1024_prefix_matches_sft512": str(l["raw_completion"]).startswith(
                str(row["baseline"]["raw_completion"])
            ),
        }
        for arm, code in codes.items():
            key = (qid, sha_string(code))
            x[f"{arm}_correct"] = cache[key]
            x[f"{arm}_resource_limit"] = failed_reason.get(key) == "candidate-resource-limit"
        details.append(x)
    result = summarize(details, iterations=iterations, seed=seed)
    result["frozen_inputs"] = frozen
    result["private_tests_accessed_only_after_freeze"] = True
    result["resource_limit_counts"] = {
        arm: sum(bool(x[f"{arm}_resource_limit"]) for x in details) for arm in ARMS
    }
    with (output_dir / "details.jsonl").open("w", encoding="utf-8") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(json.dumps({"arms": result["arms"],
                      "primary_success": result["preregistered_primary_success"]},
                     ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path("artifacts/exp004n")
    parser.add_argument("--sft512", type=Path, default=root / "sft512/runner.json")
    parser.add_argument("--base512", type=Path, default=root / "base512/runner.json")
    parser.add_argument("--sft1024", type=Path, default=root / "sft1024/runner.json")
    parser.add_argument("--public-tasks", type=Path, default=root / "data/public_tasks.jsonl")
    parser.add_argument("--manifest", type=Path, default=root / "data/manifest.json")
    parser.add_argument("--lcb-repo", type=Path, default=Path(".external/livecodebench"))
    parser.add_argument("--adapter", type=Path, default=Path("artifacts/exp006a/sft-qwen3-1.7b-lora"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output-dir", type=Path, default=root / "final_evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=20000)
    args = parser.parse_args()
    if args.source is None:
        from huggingface_hub import hf_hub_download
        source = Path(hf_hub_download(
            repo_id="livecodebench/code_generation_lite",
            filename=SOURCE_FILENAME, revision=SOURCE_REVISION, repo_type="dataset",
        ))
    else:
        source = args.source
    run_final(
        sft512_path=args.sft512, base512_path=args.base512,
        sft1024_path=args.sft1024, public_path=args.public_tasks,
        manifest_path=args.manifest, adapter_dir=args.adapter,
        lcb_repo=args.lcb_repo, source_path=source,
        output_dir=args.output_dir, seed=args.seed, iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
