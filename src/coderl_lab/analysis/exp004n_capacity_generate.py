"""Freeze Base-512 and SFT-1024 controls on independent LiveCodeBench v5.

This module never reads, decodes, exports, or evaluates private cases.
The original EXP-004L runner and frozen configuration remain unchanged.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from coderl_lab.analysis.livecodebench_official_eval import (
    OfficialLiveCodeBenchExecutor,
    evaluate_code,
    load_public_tasks,
)
from coderl_lab.datasets.livecodebench import (
    LiveCodeBenchPublicTask, build_generic_base_prompt, sha256_file,
)
from coderl_lab.analysis.local_bottleneck_intervention import _generate, _load_model


SOURCE_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
SOURCE_SHA256 = "7f77571c2a6df0c2a72a3277650309f67e01e0008e18117e624633df53f81214"
PUBLIC_SHA256 = "e695ba9fa2ce35abc8db2f3360bf711930746cd55843890177ecd518c0a4c98d"
SFT_ADAPTER_SHA256 = "e1ea9a007a3e4213cdeb1ca6b1e12d29adb83594991421757d0d40b9168d0861"
MODEL_REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
OFFICIAL_COMMIT = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
MODEL = "Qwen/Qwen3-1.7B-Base"
ARMS = ("base512", "sft1024")


def audit_input(
    *, public_path: Path, manifest_path: Path, lcb_repo: Path,
    arm: str, adapter_dir: Path, require_formal: bool = True,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError("unsupported capacity control")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if any((
        manifest.get("source_revision") != SOURCE_REVISION,
        manifest.get("source_sha256") != SOURCE_SHA256,
        manifest.get("public_view_sha256") != PUBLIC_SHA256,
        manifest.get("fine_grained_version") != "v5",
        manifest.get("source_filename") != "test5.jsonl",
        manifest.get("tasks") != 167,
        manifest.get("private_tests_exported") is not False,
        manifest.get("private_tests_decoded") is not False,
    )):
        raise ValueError("v5 public-only manifest does not match preregistration")
    if sha256_file(public_path) != PUBLIC_SHA256:
        raise ValueError("v5 public view SHA mismatch")
    proc = subprocess.run(
        ["git", "-C", str(lcb_repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    if proc.stdout.strip() != OFFICIAL_COMMIT:
        raise ValueError("official evaluator / prompt revision changed")
    public = load_public_tasks(public_path)
    if len(public) != 167:
        raise ValueError("independent v5 set must have 167 tasks")
    adapter_sha = None
    if arm == "sft1024":
        file = adapter_dir / "adapter_model.safetensors"
        if not file.is_file():
            raise FileNotFoundError("frozen EXP-006A SFT adapter is missing")
        adapter_sha = sha256_file(file)
        if adapter_sha != SFT_ADAPTER_SHA256:
            raise ValueError("EXP-006A adapter SHA mismatch")
    return {
        "experiment": "EXP-004N",
        "arm": arm,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "adapter_sha256": adapter_sha,
        "source_sha256": SOURCE_SHA256,
        "public_view_sha256": PUBLIC_SHA256,
        "official_checker_commit": OFFICIAL_COMMIT,
        "tasks": 167,
        "private_tests_accessed": False,
        "formal_dataset": require_formal,
    }


def summarize_completion(*, raw: str, token_count: int, token_cap: int) -> dict[str, Any]:
    if token_cap not in (512, 1024):
        raise ValueError("only preregistered output lengths permitted")
    code = raw.strip()
    try:
        ast.parse(code)
        syntax_ok = True
    except (SyntaxError, ValueError):
        syntax_ok = False
    return {
        "raw_completion": raw,
        "code": code,
        "generated_token_count": token_count,
        "reached_token_cap": token_count >= token_cap,
        "syntax_ok": syntax_ok,
    }


def generate_control(
    *,
    arm: str,
    public_path: Path,
    manifest_path: Path,
    lcb_repo: Path,
    adapter_dir: Path,
    output_dir: Path,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    provenance = audit_input(
        public_path=public_path, manifest_path=manifest_path,
        lcb_repo=lcb_repo, arm=arm, adapter_dir=adapter_dir,
        require_formal=max_tasks is None,
    )
    if max_tasks is not None and (max_tasks < 1 or max_tasks > 167):
        raise ValueError("smoke task count out of range")
    if (output_dir / "runner.json").exists():
        raise FileExistsError("frozen control runner exists; refusing overwrite")

    max_new_tokens = 512 if arm == "base512" else 1024
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=MODEL_REVISION, trust_remote_code=True,
    )
    if arm == "base512":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=MODEL_REVISION, dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )
        model.eval()
    else:
        model = _load_model(MODEL, MODEL_REVISION, adapter_dir)

    evaluator = OfficialLiveCodeBenchExecutor(
        livecodebench_repo=lcb_repo, timeout_seconds=6, memory_limit="1g",
    )
    public_rows = load_public_tasks(public_path)
    ids = sorted(public_rows)
    if max_tasks is not None:
        ids = ids[:max_tasks]
    frozen_rows: dict[str, Any] = {}
    with torch.inference_mode():
        for index, qid in enumerate(ids, 1):
            pub = public_rows[qid]
            task = LiveCodeBenchPublicTask(
                question_id=str(pub["question_id"]),
                question_title=str(pub["question_title"]),
                question_content=str(pub["question_content"]),
                platform=str(pub["platform"]),
                contest_id=str(pub["contest_id"]),
                contest_date=str(pub["contest_date"]),
                starter_code=str(pub["starter_code"]),
                difficulty=str(pub["difficulty"]),
                public_test_cases=tuple(pub["public_test_cases"]),
                metadata=dict(pub["metadata"]),
            )
            prompt = build_generic_base_prompt(task, livecodebench_repo=lcb_repo)
            encoded = {k: v.to(model.device) for k, v in
                       tokenizer(prompt, return_tensors="pt").items()}
            token_ids, raw, applied = _generate(
                model, encoded=encoded, tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
            )
            if applied:
                raise AssertionError("control arm cannot contain token intervention")
            completion = summarize_completion(
                raw=raw, token_count=len(token_ids), token_cap=max_new_tokens,
            )
            public_result = evaluate_code(
                executor=evaluator, task=pub, code=completion["code"],
            )
            if public_result.results and str(public_result.results[0]) in (
                "runner-error", "invalid-runner-output", "global-timeout"
            ):
                raise RuntimeError(f"public evaluator failed on {qid}")
            frozen_rows[qid] = {
                "question_id": qid, "platform": task.platform,
                "difficulty": task.difficulty,
                "public_all_pass": bool(public_result.passed),
                **completion,
            }
            if index % 10 == 0:
                print(f"EXP-004N {arm}: generated {index}/{len(ids)}", flush=True)

    result = {
        **provenance, "tasks": len(ids),
        "max_new_tokens": max_new_tokens,
        "is_formal": max_tasks is None,
        "per_task": frozen_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runner.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps({
        "experiment": "EXP-004N", "arm": arm,
        "generated": len(frozen_rows), "private_tests_accessed": False,
        "output": str(output_dir / "runner.json")
    }, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--public-tasks", type=Path, default=Path("artifacts/exp004n/data/public_tasks.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/exp004n/data/manifest.json"))
    parser.add_argument("--lcb-repo", type=Path, default=Path(".external/livecodebench"))
    parser.add_argument("--adapter", type=Path, default=Path("artifacts/exp006a/sft-qwen3-1.7b-lora"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()
    generate_control(
        arm=args.arm, public_path=args.public_tasks,
        manifest_path=args.manifest, lcb_repo=args.lcb_repo,
        adapter_dir=args.adapter, output_dir=args.output_dir,
        max_tasks=args.max_tasks,
    )


if __name__ == "__main__":
    main()
