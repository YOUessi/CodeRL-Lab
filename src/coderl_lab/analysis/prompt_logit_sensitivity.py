from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

from coderl_lab.evaluation import load_tasks
from coderl_lab.generation import build_prompt


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return float(ordered[index])


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must be non-empty")

    kl = [float(row["kl_ref_to_candidate"]) for row in rows]
    tv = [float(row["total_variation"]) for row in rows]
    centered_rms = [float(row["centered_logit_rms_delta"]) for row in rows]
    top5_jaccard = [float(row["top5_jaccard"]) for row in rows]
    ref_margin = [float(row["reference_top1_margin"]) for row in rows]
    cand_margin = [float(row["candidate_top1_margin"]) for row in rows]

    return {
        "tasks": len(rows),
        "kl_ref_to_candidate": {
            "mean": mean(kl),
            "median": median(kl),
            "p95": _quantile(kl, 0.95),
            "max": max(kl),
        },
        "total_variation": {
            "mean": mean(tv),
            "median": median(tv),
            "p95": _quantile(tv, 0.95),
            "max": max(tv),
        },
        "centered_logit_rms_delta": {
            "mean": mean(centered_rms),
            "median": median(centered_rms),
            "p95": _quantile(centered_rms, 0.95),
            "max": max(centered_rms),
        },
        "top1_agreement_fraction": (
            sum(bool(row["top1_same"]) for row in rows) / len(rows)
        ),
        "top5_jaccard": {
            "mean": mean(top5_jaccard),
            "median": median(top5_jaccard),
            "min": min(top5_jaccard),
        },
        "reference_top1_margin": {
            "mean": mean(ref_margin),
            "median": median(ref_margin),
        },
        "candidate_top1_margin": {
            "mean": mean(cand_margin),
            "median": median(cand_margin),
        },
    }


def _load_model(
    *,
    model_name: str,
    revision: str,
    adapter_path: Path,
    dtype_name: str,
):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    return model


def _prompt_logits(
    *,
    model,
    tokenizer,
    tasks,
) -> dict[str, Any]:
    import torch

    output: dict[str, Any] = {}
    with torch.inference_mode():
        for task in tasks:
            prompt = build_prompt(task.prompt, task.starter_code)
            encoded = tokenizer(prompt, return_tensors="pt")
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            logits = model(**encoded, use_cache=False).logits[0, -1].float().cpu()
            output[task.task_id] = logits
    return output


def compare_prompt_logits(
    *,
    tasks_path: Path,
    model_name: str,
    revision: str,
    reference_adapter: Path,
    candidate_adapter: Path,
    output_path: Path | None,
    max_tasks: int | None = None,
    dtype_name: str = "bfloat16",
) -> dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "prompt-logit analysis requires model dependencies; "
            "install with pip install -e '.[model]'"
        ) from exc

    tasks = list(load_tasks(tasks_path).values())
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    if not tasks:
        raise ValueError("task set is empty")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=True,
    )

    reference_model = _load_model(
        model_name=model_name,
        revision=revision,
        adapter_path=reference_adapter,
        dtype_name=dtype_name,
    )
    reference_logits = _prompt_logits(
        model=reference_model,
        tokenizer=tokenizer,
        tasks=tasks,
    )
    del reference_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    candidate_model = _load_model(
        model_name=model_name,
        revision=revision,
        adapter_path=candidate_adapter,
        dtype_name=dtype_name,
    )

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for task in tasks:
            prompt = build_prompt(task.prompt, task.starter_code)
            encoded = tokenizer(prompt, return_tensors="pt")
            encoded = {key: value.to(candidate_model.device) for key, value in encoded.items()}
            candidate = candidate_model(**encoded, use_cache=False).logits[0, -1].float()
            reference = reference_logits[task.task_id].to(candidate.device)

            ref_logp = F.log_softmax(reference, dim=-1)
            cand_logp = F.log_softmax(candidate, dim=-1)
            ref_prob = ref_logp.exp()
            cand_prob = cand_logp.exp()

            kl = float((ref_prob * (ref_logp - cand_logp)).sum().item())
            tv = float((0.5 * (ref_prob - cand_prob).abs().sum()).item())

            ref_centered = reference - reference.mean()
            cand_centered = candidate - candidate.mean()
            centered_rms = float(
                torch.sqrt(torch.mean((ref_centered - cand_centered) ** 2)).item()
            )

            ref_top5 = torch.topk(ref_prob, 5)
            cand_top5 = torch.topk(cand_prob, 5)
            ref_ids = [int(x) for x in ref_top5.indices.tolist()]
            cand_ids = [int(x) for x in cand_top5.indices.tolist()]
            union = set(ref_ids) | set(cand_ids)
            intersection = set(ref_ids) & set(cand_ids)

            ref_top2 = torch.topk(ref_prob, 2).values
            cand_top2 = torch.topk(cand_prob, 2).values
            ref_top1_id = int(ref_top5.indices[0].item())
            cand_top1_id = int(cand_top5.indices[0].item())

            rows.append(
                {
                    "task_id": task.task_id,
                    "kl_ref_to_candidate": kl,
                    "total_variation": tv,
                    "centered_logit_rms_delta": centered_rms,
                    "top1_same": ref_top1_id == cand_top1_id,
                    "reference_top1_token_id": ref_top1_id,
                    "candidate_top1_token_id": cand_top1_id,
                    "reference_top1_probability": float(ref_top5.values[0].item()),
                    "candidate_top1_probability": float(cand_top5.values[0].item()),
                    "candidate_probability_of_reference_top1": float(
                        cand_prob[ref_top1_id].item()
                    ),
                    "reference_probability_of_candidate_top1": float(
                        ref_prob[cand_top1_id].item()
                    ),
                    "reference_top1_margin": float((ref_top2[0] - ref_top2[1]).item()),
                    "candidate_top1_margin": float((cand_top2[0] - cand_top2[1]).item()),
                    "top5_jaccard": len(intersection) / len(union),
                    "reference_top5_token_ids": ref_ids,
                    "candidate_top5_token_ids": cand_ids,
                }
            )

    summary = summarize(rows)
    result = {
        "model": model_name,
        "revision": revision,
        "reference_adapter": str(reference_adapter),
        "candidate_adapter": str(candidate_adapter),
        "dtype": dtype_name,
        "summary": summary,
        "per_task": rows,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--reference-adapter", type=Path, required=True)
    parser.add_argument("--candidate-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare_prompt_logits(
        tasks_path=args.tasks,
        model_name=args.model,
        revision=args.revision,
        reference_adapter=args.reference_adapter,
        candidate_adapter=args.candidate_adapter,
        output_path=args.output,
        max_tasks=args.max_tasks,
        dtype_name=args.dtype,
    )


if __name__ == "__main__":
    main()
