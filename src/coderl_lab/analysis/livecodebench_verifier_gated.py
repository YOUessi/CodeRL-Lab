from __future__ import annotations

import argparse
import gc
import json
import subprocess
from pathlib import Path
from typing import Any

from coderl_lab.analysis.livecodebench_official_eval import (
    OfficialLiveCodeBenchExecutor,
    evaluate_code,
    load_public_tasks,
)
from coderl_lab.analysis.local_bottleneck_intervention import (
    _condition_result,
    _load_model,
    _reference_trace,
    select_intervention_positions,
)
from coderl_lab.datasets.livecodebench import (
    LiveCodeBenchPublicTask,
    build_generic_base_prompt,
)


CONDITIONS = (
    "baseline",
    "gated_low_destabilize",
    "gated_high_destabilize",
    "always_low_destabilize",
)


def _task_from_public_dict(row: dict[str, Any]) -> LiveCodeBenchPublicTask:
    return LiveCodeBenchPublicTask(
        question_id=str(row["question_id"]),
        question_title=str(row["question_title"]),
        question_content=str(row["question_content"]),
        platform=str(row["platform"]),
        contest_id=str(row["contest_id"]),
        contest_date=str(row["contest_date"]),
        starter_code=str(row.get("starter_code", "")),
        difficulty=str(row["difficulty"]),
        public_test_cases=tuple(row["public_test_cases"]),
        metadata=dict(row.get("metadata", {})),
    )


def _baseline_result(raw: str) -> dict[str, Any]:
    return {
        "raw_completion": raw,
        "code": raw.strip(),
        "exact_match_reference": True,
        "first_divergence_index": None,
        "intervention_applied": False,
        "target_position": None,
        "bias": 0.0,
    }


def _with_code(result: dict[str, Any]) -> dict[str, Any]:
    out = dict(result)
    out["code"] = str(out.get("raw_completion", "")).strip()
    return out


def _pinned_source_sha(repo: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "cannot read pinned LiveCodeBench source SHA: "
            + (proc.stderr.strip() or proc.stdout.strip())
        )
    return proc.stdout.strip()


def write_condition_predictions(
    *,
    per_task: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for condition in CONDITIONS:
        path = output_dir / f"{condition}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for question_id in sorted(per_task):
                row = per_task[question_id]
                condition_row = row[condition]
                handle.write(
                    json.dumps(
                        {
                            "question_id": question_id,
                            "sample_id": 0,
                            "completion": str(condition_row["code"]),
                            "raw_completion": str(
                                condition_row["raw_completion"]
                            ),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def run_livecodebench_verifier_gated(
    *,
    public_tasks_path: Path,
    model_name: str,
    revision: str,
    adapter: Path,
    livecodebench_repo: Path,
    output_dir: Path,
    docker_image: str = "coderl-lab-lcb-eval",
    max_new_tokens: int = 512,
    primary_window: int = 128,
    low_threshold: float = 0.05,
    high_threshold: float = 0.20,
    bias: float = 0.25,
    public_timeout: int = 6,
    public_memory: str = "1g",
    max_tasks: int | None = None,
) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "LiveCodeBench verifier-gated runner requires model dependencies"
        ) from exc

    public_map = load_public_tasks(public_tasks_path)
    question_ids = sorted(public_map)
    if max_tasks is not None:
        question_ids = question_ids[:max_tasks]

    official_sha = _pinned_source_sha(livecodebench_repo)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=True,
    )
    model = _load_model(model_name, revision, adapter)
    evaluator = OfficialLiveCodeBenchExecutor(
        livecodebench_repo=livecodebench_repo,
        docker_image=docker_image,
        timeout_seconds=public_timeout,
        memory_limit=public_memory,
    )

    per_task: dict[str, Any] = {}
    eligible_tasks = 0
    gate_triggered_tasks = 0
    high_control_available_tasks = 0
    public_pass_tasks = 0

    with torch.inference_mode():
        for question_id in question_ids:
            public_row = public_map[question_id]
            task = _task_from_public_dict(public_row)
            prompt = build_generic_base_prompt(
                task,
                livecodebench_repo=livecodebench_repo,
            )

            encoded = tokenizer(prompt, return_tensors="pt")
            encoded = {
                key: value.to(model.device)
                for key, value in encoded.items()
            }

            ref_tokens, ref_raw, margins = _reference_trace(
                model,
                encoded=encoded,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
            )
            baseline = _baseline_result(ref_raw)

            public_result = evaluate_code(
                executor=evaluator,
                task=public_row,
                code=baseline["code"],
            )
            public_all_pass = bool(public_result.passed)
            public_pass_tasks += int(public_all_pass)

            low_position, high_position = select_intervention_positions(
                margins,
                primary_window=primary_window,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
            )
            eligible = low_position is not None
            eligible_tasks += int(eligible)
            high_control_available_tasks += int(high_position is not None)

            gate = bool((not public_all_pass) and eligible)
            gate_triggered_tasks += int(gate)

            if eligible:
                always_low = _with_code(
                    _condition_result(
                        model,
                        encoded=encoded,
                        tokenizer=tokenizer,
                        reference_tokens=ref_tokens,
                        reference_raw=ref_raw,
                        target_position=low_position,
                        bias=-abs(bias),
                        max_new_tokens=max_new_tokens,
                    )
                )
            else:
                always_low = dict(baseline)

            if gate:
                gated_low = dict(always_low)
                if high_position is not None:
                    gated_high = _with_code(
                        _condition_result(
                            model,
                            encoded=encoded,
                            tokenizer=tokenizer,
                            reference_tokens=ref_tokens,
                            reference_raw=ref_raw,
                            target_position=high_position,
                            bias=-abs(bias),
                            max_new_tokens=max_new_tokens,
                        )
                    )
                else:
                    gated_high = dict(baseline)
            else:
                gated_low = dict(baseline)
                gated_high = dict(baseline)

            per_task[question_id] = {
                "question_id": question_id,
                "platform": task.platform,
                "difficulty": task.difficulty,
                "contest_date": task.contest_date,
                "reference_token_count": len(ref_tokens),
                "eligible": eligible,
                "low_position": low_position,
                "low_margin": (
                    margins[low_position]
                    if low_position is not None
                    else None
                ),
                "high_control_position": high_position,
                "high_control_margin": (
                    margins[high_position]
                    if high_position is not None
                    else None
                ),
                "public_all_pass": public_all_pass,
                "public_results": list(public_result.results),
                "gate_triggered": gate,
                "baseline": baseline,
                "gated_low_destabilize": gated_low,
                "gated_high_destabilize": gated_high,
                "always_low_destabilize": always_low,
            }

    result = {
        "experiment": "EXP-004L",
        "model": model_name,
        "revision": revision,
        "adapter": str(adapter),
        "tasks": len(question_ids),
        "official_livecodebench_commit": official_sha,
        "prompt_style": "GenericBase official one-shot",
        "max_new_tokens": max_new_tokens,
        "primary_window": primary_window,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "bias": bias,
        "eligible_tasks": eligible_tasks,
        "gate_triggered_tasks": gate_triggered_tasks,
        "high_control_available_tasks": high_control_available_tasks,
        "public_pass_tasks": public_pass_tasks,
        "private_tests_accessed": False,
        "per_task": per_task,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runner.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_condition_predictions(
        per_task=per_task,
        output_dir=output_dir / "predictions",
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        json.dumps(
            {
                "tasks": result["tasks"],
                "eligible_tasks": eligible_tasks,
                "gate_triggered_tasks": gate_triggered_tasks,
                "public_pass_tasks": public_pass_tasks,
                "private_tests_accessed": False,
                "official_livecodebench_commit": official_sha,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen verifier-gated bottleneck escape on LiveCodeBench"
    )
    parser.add_argument("--public-tasks", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument(
        "--livecodebench-repo",
        type=Path,
        default=Path(".external/livecodebench"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--docker-image", default="coderl-lab-lcb-eval")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--primary-window", type=int, default=128)
    parser.add_argument("--low-threshold", type=float, default=0.05)
    parser.add_argument("--high-threshold", type=float, default=0.20)
    parser.add_argument("--bias", type=float, default=0.25)
    parser.add_argument("--public-timeout", type=int, default=6)
    parser.add_argument("--public-memory", default="1g")
    parser.add_argument("--max-tasks", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_livecodebench_verifier_gated(
        public_tasks_path=args.public_tasks,
        model_name=args.model,
        revision=args.revision,
        adapter=args.adapter,
        livecodebench_repo=args.livecodebench_repo,
        output_dir=args.output_dir,
        docker_image=args.docker_image,
        max_new_tokens=args.max_new_tokens,
        primary_window=args.primary_window,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        bias=args.bias,
        public_timeout=args.public_timeout,
        public_memory=args.public_memory,
        max_tasks=args.max_tasks,
    )


if __name__ == "__main__":
    main()
