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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return float(ordered[index])


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"divergent_tasks": 0}

    positions = [int(row["divergence_index"]) for row in rows]
    kl = [float(row["kl_ref_to_candidate"]) for row in rows]
    tv = [float(row["total_variation"]) for row in rows]
    ref_margin = [float(row["reference_top1_margin"]) for row in rows]
    cand_margin = [float(row["candidate_top1_margin"]) for row in rows]
    ref_competing_gap = [
        float(row["reference_logit_gap_ref_minus_candidate_token"])
        for row in rows
    ]
    cand_competing_gap = [
        float(row["candidate_logit_gap_candidate_minus_ref_token"])
        for row in rows
    ]

    return {
        "divergent_tasks": len(rows),
        "divergence_index": {
            "mean": mean(positions),
            "median": median(positions),
            "p25": _quantile([float(x) for x in positions], 0.25),
            "p75": _quantile([float(x) for x in positions], 0.75),
            "min": min(positions),
            "max": max(positions),
        },
        "diverge_within_first_10_tokens_fraction": (
            sum(pos < 10 for pos in positions) / len(positions)
        ),
        "diverge_within_first_50_tokens_fraction": (
            sum(pos < 50 for pos in positions) / len(positions)
        ),
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
        "reference_top1_margin": {
            "mean": mean(ref_margin),
            "median": median(ref_margin),
        },
        "candidate_top1_margin": {
            "mean": mean(cand_margin),
            "median": median(cand_margin),
        },
        "reference_competing_token_gap": {
            "mean": mean(ref_competing_gap),
            "median": median(ref_competing_gap),
        },
        "candidate_competing_token_gap": {
            "mean": mean(cand_competing_gap),
            "median": median(cand_competing_gap),
        },
    }


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


def analyze(
    *,
    tasks_path: Path,
    model_name: str,
    revision: str,
    reference_adapter: Path,
    candidate_adapter: Path,
    reference_predictions: Path,
    candidate_predictions: Path,
    output_path: Path | None,
    max_new_tokens: int = 512,
) -> dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "greedy-divergence analysis requires model dependencies"
        ) from exc

    reference_rows = {
        (str(row["task_id"]), int(row["sample_id"])): row
        for row in load_jsonl(reference_predictions)
    }
    candidate_rows = {
        (str(row["task_id"]), int(row["sample_id"])): row
        for row in load_jsonl(candidate_predictions)
    }
    changed_task_ids = sorted(
        {
            task_id
            for (task_id, sample_id), row in reference_rows.items()
            if sample_id == 0
            and (task_id, sample_id) in candidate_rows
            and str(row["raw_completion"])
            != str(candidate_rows[(task_id, sample_id)]["raw_completion"])
        }
    )

    task_map = load_tasks(tasks_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=True,
    )
    reference_model = _load_model(model_name, revision, reference_adapter)
    candidate_model = _load_model(model_name, revision, candidate_adapter)

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for task_id in changed_task_ids:
            task = task_map[task_id]
            prompt = build_prompt(task.prompt, task.starter_code)
            encoded = tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(reference_model.device)
            attention_mask = encoded["attention_mask"].to(reference_model.device)
            prompt_length = input_ids.shape[1]

            ref_output = reference_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )[0, prompt_length:].tolist()

            cand_input_ids = encoded["input_ids"].to(candidate_model.device)
            cand_attention_mask = encoded["attention_mask"].to(candidate_model.device)
            cand_output = candidate_model.generate(
                input_ids=cand_input_ids,
                attention_mask=cand_attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )[0, prompt_length:].tolist()

            divergence = first_divergence(ref_output, cand_output)
            if divergence is None:
                continue
            if divergence >= len(ref_output) or divergence >= len(cand_output):
                rows.append(
                    {
                        "task_id": task_id,
                        "divergence_index": divergence,
                        "length_only_divergence": True,
                    }
                )
                continue

            common = ref_output[:divergence]
            ref_prefix = torch.tensor(
                [encoded["input_ids"][0].tolist() + common],
                device=reference_model.device,
            )
            cand_prefix = ref_prefix.to(candidate_model.device)

            ref_logits = reference_model(
                input_ids=ref_prefix,
                use_cache=False,
            ).logits[0, -1].float()
            cand_logits = candidate_model(
                input_ids=cand_prefix,
                use_cache=False,
            ).logits[0, -1].float()

            ref_logp = F.log_softmax(ref_logits, dim=-1)
            cand_logp = F.log_softmax(cand_logits, dim=-1)
            ref_prob = ref_logp.exp()
            cand_prob = cand_logp.exp()

            ref_token = int(ref_output[divergence])
            cand_token = int(cand_output[divergence])
            ref_top2 = torch.topk(ref_prob, 2).values
            cand_top2 = torch.topk(cand_prob, 2).values

            rows.append(
                {
                    "task_id": task_id,
                    "divergence_index": divergence,
                    "length_only_divergence": False,
                    "reference_token_id": ref_token,
                    "candidate_token_id": cand_token,
                    "reference_token_text": tokenizer.decode([ref_token]),
                    "candidate_token_text": tokenizer.decode([cand_token]),
                    "kl_ref_to_candidate": float(
                        (ref_prob * (ref_logp - cand_logp)).sum().item()
                    ),
                    "total_variation": float(
                        (0.5 * (ref_prob - cand_prob).abs().sum()).item()
                    ),
                    "reference_top1_margin": float(
                        (ref_top2[0] - ref_top2[1]).item()
                    ),
                    "candidate_top1_margin": float(
                        (cand_top2[0] - cand_top2[1]).item()
                    ),
                    "reference_probability_of_reference_token": float(
                        ref_prob[ref_token].item()
                    ),
                    "reference_probability_of_candidate_token": float(
                        ref_prob[cand_token].item()
                    ),
                    "candidate_probability_of_candidate_token": float(
                        cand_prob[cand_token].item()
                    ),
                    "candidate_probability_of_reference_token": float(
                        cand_prob[ref_token].item()
                    ),
                    "reference_logit_gap_ref_minus_candidate_token": float(
                        (ref_logits[ref_token] - ref_logits[cand_token]).item()
                    ),
                    "candidate_logit_gap_candidate_minus_ref_token": float(
                        (cand_logits[cand_token] - cand_logits[ref_token]).item()
                    ),
                }
            )

    numeric_rows = [
        row for row in rows if not bool(row.get("length_only_divergence", False))
    ]
    result = {
        "changed_tasks_from_prediction_files": len(changed_task_ids),
        "regenerated_divergent_tasks": len(rows),
        "numeric_divergence_tasks": len(numeric_rows),
        "summary": summarize(numeric_rows),
        "per_task": rows,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))

    del reference_model
    del candidate_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--reference-adapter", type=Path, required=True)
    parser.add_argument("--candidate-adapter", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze(
        tasks_path=args.tasks,
        model_name=args.model,
        revision=args.revision,
        reference_adapter=args.reference_adapter,
        candidate_adapter=args.candidate_adapter,
        reference_predictions=args.reference_predictions,
        candidate_predictions=args.candidate_predictions,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
