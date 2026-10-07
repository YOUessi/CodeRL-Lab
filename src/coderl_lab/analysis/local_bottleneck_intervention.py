from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

from coderl_lab.evaluation import load_tasks
from coderl_lab.generation import build_prompt
from coderl_lab.analysis.greedy_first_divergence import first_divergence


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_intervention_positions(
    probability_margins: list[float],
    *,
    primary_window: int = 128,
    low_threshold: float = 0.05,
    high_threshold: float = 0.20,
) -> tuple[int | None, int | None]:
    window = min(primary_window, len(probability_margins))
    low_positions = [
        i for i, value in enumerate(probability_margins[:window])
        if value <= low_threshold
    ]
    if not low_positions:
        return None, None

    low_position = low_positions[0]
    high_positions = [
        i for i, value in enumerate(probability_margins[:window])
        if value >= high_threshold
    ]
    high_position = (
        min(high_positions, key=lambda i: (abs(i - low_position), i))
        if high_positions
        else None
    )
    return low_position, high_position


def _trim_special(token_ids: list[int], special_ids: set[int]) -> list[int]:
    values = list(token_ids)
    while values and values[-1] in special_ids:
        values.pop()
    return values


def _load_model(model_name: str, revision: str, adapter: Path):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter))
    model.eval()
    return model


class _SinglePositionReferenceBias:
    def __init__(
        self,
        *,
        prompt_length: int,
        target_position: int | None,
        reference_tokens: list[int],
        bias: float,
    ) -> None:
        self.prompt_length = prompt_length
        self.target_position = target_position
        self.reference_tokens = reference_tokens
        self.bias = float(bias)
        self.applied = False

    def __call__(self, input_ids, scores):
        if self.target_position is None or self.applied:
            return scores

        generated = input_ids[0, self.prompt_length :].tolist()
        step = len(generated)
        if step != self.target_position:
            return scores
        if generated != self.reference_tokens[:step]:
            return scores
        if step >= len(self.reference_tokens):
            return scores

        output = scores.clone()
        output[0, self.reference_tokens[step]] += self.bias
        self.applied = True
        return output


def _generate(
    model,
    *,
    encoded: dict[str, Any],
    tokenizer,
    max_new_tokens: int,
    reference_tokens: list[int] | None = None,
    target_position: int | None = None,
    bias: float = 0.0,
):
    from transformers import LogitsProcessorList

    prompt_length = encoded["input_ids"].shape[1]
    processor = None
    kwargs: dict[str, Any] = {}
    if reference_tokens is not None and target_position is not None:
        processor = _SinglePositionReferenceBias(
            prompt_length=prompt_length,
            target_position=target_position,
            reference_tokens=reference_tokens,
            bias=bias,
        )
        kwargs["logits_processor"] = LogitsProcessorList([processor])

    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        **kwargs,
    )[0, prompt_length:].tolist()
    special = {
        x
        for x in (tokenizer.eos_token_id, tokenizer.pad_token_id)
        if x is not None
    }
    tokens = _trim_special(output, special)
    raw = tokenizer.decode(tokens, skip_special_tokens=True)
    return tokens, raw, bool(processor.applied if processor is not None else False)


def _reference_trace(
    reference_model,
    *,
    encoded: dict[str, Any],
    tokenizer,
    max_new_tokens: int,
):
    import torch

    tokens, raw, _ = _generate(
        reference_model,
        encoded=encoded,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )
    if not tokens:
        return tokens, raw, []

    prompt_ids = encoded["input_ids"][0].tolist()
    full_ids = torch.tensor(
        [prompt_ids + tokens],
        device=reference_model.device,
    )
    logits = reference_model(
        input_ids=full_ids,
        use_cache=False,
    ).logits[0]
    prompt_length = len(prompt_ids)
    pred_logits = logits[
        prompt_length - 1 : prompt_length - 1 + len(tokens)
    ].float()
    top2 = torch.topk(pred_logits, k=2, dim=-1)
    log_norm = torch.logsumexp(pred_logits, dim=-1)
    top2_prob = torch.exp(top2.values - log_norm[:, None])
    margins = (top2_prob[:, 0] - top2_prob[:, 1]).cpu().tolist()
    return tokens, raw, [float(x) for x in margins]


def _condition_result(
    candidate_model,
    *,
    encoded: dict[str, Any],
    tokenizer,
    reference_tokens: list[int],
    reference_raw: str,
    target_position: int | None,
    bias: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    tokens, raw, applied = _generate(
        candidate_model,
        encoded=encoded,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        reference_tokens=reference_tokens,
        target_position=target_position,
        bias=bias,
    )
    divergence = first_divergence(reference_tokens, tokens)
    return {
        "raw_completion": raw,
        "exact_match_reference": raw == reference_raw,
        "first_divergence_index": divergence,
        "intervention_applied": applied,
        "target_position": target_position,
        "bias": bias,
    }


def run_interventions(
    *,
    tasks_path: Path,
    model_name: str,
    revision: str,
    reference_adapter: Path,
    candidate_adapter: Path,
    reference_predictions: Path,
    baseline_predictions: Path,
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
        raise RuntimeError("EXP-004H requires model dependencies") from exc

    tasks = list(load_tasks(tasks_path).values())
    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    reference_rows = {
        str(row["task_id"]): row
        for row in load_jsonl(reference_predictions)
        if int(row["sample_id"]) == 0
    }
    baseline_rows = {
        str(row["task_id"]): row
        for row in load_jsonl(baseline_predictions)
        if int(row["sample_id"]) == 0
    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=True,
    )
    reference_model = _load_model(model_name, revision, reference_adapter)
    candidate_model = _load_model(model_name, revision, candidate_adapter)

    per_task: dict[str, Any] = {}
    reference_reproduced = 0
    baseline_reproduced = 0

    with torch.inference_mode():
        for task in tasks:
            prompt = build_prompt(task.prompt, task.starter_code)
            encoded_ref = tokenizer(prompt, return_tensors="pt")
            encoded_ref = {
                k: v.to(reference_model.device)
                for k, v in encoded_ref.items()
            }
            ref_tokens, ref_raw, margins = _reference_trace(
                reference_model,
                encoded=encoded_ref,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
            )

            stored_ref = reference_rows.get(task.task_id)
            ref_matches = (
                stored_ref is not None
                and str(stored_ref.get("raw_completion", "")) == ref_raw
            )
            reference_reproduced += int(ref_matches)

            low_position, high_position = select_intervention_positions(
                margins,
                primary_window=primary_window,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
            )

            encoded_cand = tokenizer(prompt, return_tensors="pt")
            encoded_cand = {
                k: v.to(candidate_model.device)
                for k, v in encoded_cand.items()
            }
            baseline_tokens, baseline_raw, _ = _generate(
                candidate_model,
                encoded=encoded_cand,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
            )
            stored_base = baseline_rows.get(task.task_id)
            baseline_matches_file = (
                stored_base is not None
                and str(stored_base.get("raw_completion", ""))
                == baseline_raw
            )
            baseline_reproduced += int(baseline_matches_file)

            baseline_div = first_divergence(ref_tokens, baseline_tokens)
            row: dict[str, Any] = {
                "task_id": task.task_id,
                "reference_reproduced": ref_matches,
                "baseline_reproduced": baseline_matches_file,
                "eligible": low_position is not None,
                "reference_token_count": len(ref_tokens),
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
                "baseline": {
                    "raw_completion": baseline_raw,
                    "exact_match_reference": baseline_raw == ref_raw,
                    "first_divergence_index": baseline_div,
                },
            }

            row["low_stabilize"] = _condition_result(
                candidate_model,
                encoded=encoded_cand,
                tokenizer=tokenizer,
                reference_tokens=ref_tokens,
                reference_raw=ref_raw,
                target_position=low_position,
                bias=abs(bias),
                max_new_tokens=max_new_tokens,
            )
            row["low_destabilize"] = _condition_result(
                candidate_model,
                encoded=encoded_cand,
                tokenizer=tokenizer,
                reference_tokens=ref_tokens,
                reference_raw=ref_raw,
                target_position=low_position,
                bias=-abs(bias),
                max_new_tokens=max_new_tokens,
            )
            row["high_stabilize"] = _condition_result(
                candidate_model,
                encoded=encoded_cand,
                tokenizer=tokenizer,
                reference_tokens=ref_tokens,
                reference_raw=ref_raw,
                target_position=high_position,
                bias=abs(bias),
                max_new_tokens=max_new_tokens,
            )

            per_task[task.task_id] = row

    result = {
        "model": model_name,
        "revision": revision,
        "reference_adapter": str(reference_adapter),
        "candidate_adapter": str(candidate_adapter),
        "tasks": len(tasks),
        "primary_window": primary_window,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "bias": bias,
        "reference_reproduction": {
            "matched": reference_reproduced,
            "total": len(tasks),
        },
        "baseline_reproduction": {
            "matched": baseline_reproduced,
            "total": len(tasks),
        },
        "eligible_tasks": sum(
            bool(row["eligible"]) for row in per_task.values()
        ),
        "per_task": per_task,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    del reference_model, candidate_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        json.dumps(
            {
                "tasks": result["tasks"],
                "eligible_tasks": result["eligible_tasks"],
                "reference_reproduction": result["reference_reproduction"],
                "baseline_reproduction": result["baseline_reproduction"],
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
    parser.add_argument("--reference-adapter", type=Path, required=True)
    parser.add_argument("--candidate-adapter", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
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
    run_interventions(
        tasks_path=args.tasks,
        model_name=args.model,
        revision=args.revision,
        reference_adapter=args.reference_adapter,
        candidate_adapter=args.candidate_adapter,
        reference_predictions=args.reference_predictions,
        baseline_predictions=args.baseline_predictions,
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
