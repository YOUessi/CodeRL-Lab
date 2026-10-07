from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

from coderl_lab.analysis.greedy_first_divergence import first_divergence
from coderl_lab.analysis.local_bottleneck_intervention import (
    _condition_result,
    _load_model,
    _reference_trace,
    load_jsonl,
    select_intervention_positions,
)
from coderl_lab.evaluation import load_tasks
from coderl_lab.generation import build_prompt


def load_reference_eval(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row["task_id"])
        if task_id in out:
            raise ValueError(f"duplicate reference eval task: {task_id}")
        out[task_id] = row
    return out


def _baseline_result(raw: str) -> dict[str, Any]:
    return {
        "raw_completion": raw,
        "exact_match_reference": True,
        "first_divergence_index": None,
        "intervention_applied": False,
        "target_position": None,
        "bias": 0.0,
    }


def run_verifier_gated_escape(
    *,
    tasks_path: Path,
    model_name: str,
    revision: str,
    adapter: Path,
    reference_predictions: Path,
    reference_details: Path,
    output_path: Path,
    max_new_tokens: int = 512,
    primary_window: int = 128,
    low_threshold: float = 0.05,
    high_threshold: float = 0.20,
    bias: float = 0.25,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("EXP-004J requires model dependencies") from exc

    tasks = list(load_tasks(tasks_path).values())
    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    stored_reference = {
        str(row["task_id"]): row
        for row in load_jsonl(reference_predictions)
        if int(row["sample_id"]) == 0
    }
    stored_eval = load_reference_eval(reference_details)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=True,
    )
    model = _load_model(model_name, revision, adapter)

    per_task: dict[str, Any] = {}
    reference_reproduced = 0
    gate_triggered = 0
    eligible_tasks = 0
    high_control_available = 0

    with torch.inference_mode():
        for task in tasks:
            prompt = build_prompt(task.prompt, task.starter_code)
            encoded = tokenizer(prompt, return_tensors="pt")
            encoded = {k: v.to(model.device) for k, v in encoded.items()}

            ref_tokens, ref_raw, margins = _reference_trace(
                model,
                encoded=encoded,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
            )
            stored = stored_reference.get(task.task_id)
            reproduced = (
                stored is not None
                and str(stored.get("raw_completion", "")) == ref_raw
            )
            reference_reproduced += int(reproduced)

            low_position, high_position = select_intervention_positions(
                margins,
                primary_window=primary_window,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
            )
            eligible = low_position is not None
            eligible_tasks += int(eligible)
            high_control_available += int(high_position is not None)

            eval_row = stored_eval[task.task_id]
            public_pass_rate = float(eval_row["public"]["pass_rate"])
            public_all_pass = public_pass_rate == 1.0
            public_fail = not public_all_pass
            gate = bool(public_fail and eligible)
            gate_triggered += int(gate)

            baseline = _baseline_result(ref_raw)

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
                "reference_reproduced": reproduced,
                "reference_token_count": len(ref_tokens),
                "public_pass_rate": public_pass_rate,
                "public_all_pass": public_all_pass,
                "gate_triggered": gate,
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
                "baseline": baseline,
                "gated_low_destabilize": gated_low,
                "gated_high_destabilize": gated_high,
                "always_low_destabilize": always_low,
            }

    result = {
        "model": model_name,
        "revision": revision,
        "adapter": str(adapter),
        "tasks": len(tasks),
        "primary_window": primary_window,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "bias": bias,
        "reference_reproduction": {
            "matched": reference_reproduced,
            "total": len(tasks),
        },
        "eligible_tasks": eligible_tasks,
        "gate_triggered_tasks": gate_triggered,
        "high_control_available_tasks": high_control_available,
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
                "tasks": result["tasks"],
                "reference_reproduction": result["reference_reproduction"],
                "eligible_tasks": result["eligible_tasks"],
                "gate_triggered_tasks": result["gate_triggered_tasks"],
                "high_control_available_tasks": result[
                    "high_control_available_tasks"
                ],
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
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--reference-details", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--primary-window", type=int, default=128)
    parser.add_argument("--low-threshold", type=float, default=0.05)
    parser.add_argument("--high-threshold", type=float, default=0.20)
    parser.add_argument("--bias", type=float, default=0.25)
    parser.add_argument("--max-tasks", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_verifier_gated_escape(
        tasks_path=args.tasks,
        model_name=args.model,
        revision=args.revision,
        adapter=args.adapter,
        reference_predictions=args.reference_predictions,
        reference_details=args.reference_details,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
        primary_window=args.primary_window,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        bias=args.bias,
        max_tasks=args.max_tasks,
    )


if __name__ == "__main__":
    main()
