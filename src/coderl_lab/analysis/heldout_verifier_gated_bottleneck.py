from __future__ import annotations

import argparse
import gc
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from coderl_lab.analysis.local_bottleneck_intervention import (
    _condition_result,
    _load_model,
    _reference_trace,
    select_intervention_positions,
)
from coderl_lab.evaluation import load_tasks
from coderl_lab.execution import PythonExecutor, report_to_dict
from coderl_lab.generation import build_prompt, extract_python_code


def should_trigger_gate(
    *,
    public_pass_rate: float,
    low_position: int | None,
) -> bool:
    return bool(public_pass_rate < 1.0 and low_position is not None)


def _baseline_result(raw: str) -> dict[str, Any]:
    return {
        "raw_completion": raw,
        "exact_match_reference": True,
        "first_divergence_index": None,
        "intervention_applied": False,
        "target_position": None,
        "bias": 0.0,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_heldout(
    *,
    tasks_path: Path,
    model_name: str,
    revision: str,
    adapter: Path,
    output_path: Path,
    baseline_predictions_path: Path,
    public_details_path: Path,
    max_new_tokens: int = 512,
    primary_window: int = 128,
    low_threshold: float = 0.05,
    high_threshold: float = 0.20,
    bias: float = 0.25,
    public_workers: int = 4,
    public_timeout: float = 5.0,
    public_memory: str = "512m",
    max_tasks: int | None = None,
) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("EXP-004K requires model dependencies") from exc

    tasks = list(load_tasks(tasks_path).values())
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    task_map = {task.task_id: task for task in tasks}

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=True,
    )
    model = _load_model(model_name, revision, adapter)

    traces: dict[str, dict[str, Any]] = {}
    baseline_rows: list[dict[str, Any]] = []

    # Phase 1: model-only deterministic reference generation.
    # No public or hidden tests are touched here.
    with torch.inference_mode():
        for task in tasks:
            prompt = build_prompt(task.prompt, task.starter_code)
            encoded = tokenizer(prompt, return_tensors="pt")
            encoded = {k: v.to(model.device) for k, v in encoded.items()}

            tokens, raw, margins = _reference_trace(
                model,
                encoded=encoded,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
            )
            low_position, high_position = select_intervention_positions(
                margins,
                primary_window=primary_window,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
            )

            traces[task.task_id] = {
                "reference_tokens": tokens,
                "reference_raw": raw,
                "reference_token_count": len(tokens),
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
            }
            baseline_rows.append(
                {
                    "task_id": task.task_id,
                    "sample_id": 0,
                    "completion": extract_python_code(
                        raw,
                        task.entry_point,
                    ),
                    "raw_completion": raw,
                    "generation": {
                        "model": model_name,
                        "revision": revision,
                        "adapter": str(adapter),
                        "greedy": True,
                    },
                }
            )

    _write_jsonl(baseline_predictions_path, baseline_rows)

    # Phase 2: PUBLIC-ONLY evaluation. Hidden tests are never passed here.
    executor = PythonExecutor(
        mode="docker",
        timeout_seconds=public_timeout,
        memory_limit=public_memory,
    )

    def public_worker(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        task_id = str(row["task_id"])
        task = task_map[task_id]
        report = executor.run(
            str(row["completion"]),
            entry_point=task.entry_point,
            cases=task.public_tests,
            setup_code=task.setup_code,
        )
        return task_id, report_to_dict(report)

    with ThreadPoolExecutor(max_workers=public_workers) as pool:
        public_results = dict(pool.map(public_worker, baseline_rows))

    public_rows = [
        {
            "task_id": task.task_id,
            "public": public_results[task.task_id],
            "public_all_pass": (
                float(public_results[task.task_id]["pass_rate"]) == 1.0
            ),
        }
        for task in tasks
    ]
    _write_jsonl(public_details_path, public_rows)

    # Phase 3: gate using public-only results, then perform second-pass decoding.
    per_task: dict[str, Any] = {}
    eligible_tasks = 0
    gate_triggered_tasks = 0
    high_control_available_tasks = 0

    with torch.inference_mode():
        for task in tasks:
            trace = traces[task.task_id]
            ref_tokens = list(trace["reference_tokens"])
            ref_raw = str(trace["reference_raw"])
            low_position = trace["low_position"]
            high_position = trace["high_control_position"]
            public_report = public_results[task.task_id]
            public_pass_rate = float(public_report["pass_rate"])
            public_all_pass = public_pass_rate == 1.0

            eligible = low_position is not None
            gate = should_trigger_gate(
                public_pass_rate=public_pass_rate,
                low_position=low_position,
            )
            eligible_tasks += int(eligible)
            gate_triggered_tasks += int(gate)
            high_control_available_tasks += int(high_position is not None)

            baseline = _baseline_result(ref_raw)

            prompt = build_prompt(task.prompt, task.starter_code)
            encoded = tokenizer(prompt, return_tensors="pt")
            encoded = {k: v.to(model.device) for k, v in encoded.items()}

            if eligible:
                always_low = _condition_result(
                    model,
                    encoded=encoded,
                    tokenizer=tokenizer,
                    reference_tokens=ref_tokens,
                    reference_raw=ref_raw,
                    target_position=low_position,
                    bias=-abs(bias),
                    max_new_tokens=max_new_tokens,
                )
            else:
                always_low = dict(baseline)

            if gate:
                gated_low = dict(always_low)
                if high_position is not None:
                    gated_high = _condition_result(
                        model,
                        encoded=encoded,
                        tokenizer=tokenizer,
                        reference_tokens=ref_tokens,
                        reference_raw=ref_raw,
                        target_position=high_position,
                        bias=-abs(bias),
                        max_new_tokens=max_new_tokens,
                    )
                else:
                    gated_high = dict(baseline)
            else:
                gated_low = dict(baseline)
                gated_high = dict(baseline)

            per_task[task.task_id] = {
                "task_id": task.task_id,
                "reference_token_count": int(
                    trace["reference_token_count"]
                ),
                "public_pass_rate": public_pass_rate,
                "public_all_pass": public_all_pass,
                "gate_triggered": gate,
                "eligible": eligible,
                "low_position": low_position,
                "low_margin": trace["low_margin"],
                "high_control_position": high_position,
                "high_control_margin": trace["high_control_margin"],
                "baseline": baseline,
                "gated_low_destabilize": gated_low,
                "gated_high_destabilize": gated_high,
                "always_low_destabilize": always_low,
            }

    result = {
        "experiment": "EXP-004K",
        "model": model_name,
        "revision": revision,
        "adapter": str(adapter),
        "tasks": len(tasks),
        "primary_window": primary_window,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "bias": bias,
        "eligible_tasks": eligible_tasks,
        "gate_triggered_tasks": gate_triggered_tasks,
        "high_control_available_tasks": high_control_available_tasks,
        "hidden_tests_accessed": False,
        "public_gate_artifact": str(public_details_path),
        "baseline_predictions": str(baseline_predictions_path),
        "per_task": per_task,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        json.dumps(
            {
                "tasks": len(tasks),
                "eligible_tasks": eligible_tasks,
                "gate_triggered_tasks": gate_triggered_tasks,
                "high_control_available_tasks": (
                    high_control_available_tasks
                ),
                "hidden_tests_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--public-details", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--primary-window", type=int, default=128)
    parser.add_argument("--low-threshold", type=float, default=0.05)
    parser.add_argument("--high-threshold", type=float, default=0.20)
    parser.add_argument("--bias", type=float, default=0.25)
    parser.add_argument("--public-workers", type=int, default=4)
    parser.add_argument("--public-timeout", type=float, default=5.0)
    parser.add_argument("--public-memory", default="512m")
    parser.add_argument("--max-tasks", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_heldout(
        tasks_path=args.tasks,
        model_name=args.model,
        revision=args.revision,
        adapter=args.adapter,
        output_path=args.output,
        baseline_predictions_path=args.baseline_predictions,
        public_details_path=args.public_details,
        max_new_tokens=args.max_new_tokens,
        primary_window=args.primary_window,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        bias=args.bias,
        public_workers=args.public_workers,
        public_timeout=args.public_timeout,
        public_memory=args.public_memory,
        max_tasks=args.max_tasks,
    )


if __name__ == "__main__":
    main()
