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


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    weight = pos - lo
    return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)


def _margin_summary(
    probability_margins: list[float],
    logit_margins: list[float],
) -> dict[str, Any]:
    if not probability_margins:
        return {
            "token_count": 0,
            "probability_margin": {},
            "logit_margin": {},
        }

    return {
        "token_count": len(probability_margins),
        "probability_margin": {
            "mean": mean(probability_margins),
            "median": median(probability_margins),
            "min": min(probability_margins),
            "p10": _quantile(probability_margins, 0.10),
            "p25": _quantile(probability_margins, 0.25),
            "fraction_le_1e_6": (
                sum(x <= 1e-6 for x in probability_margins)
                / len(probability_margins)
            ),
            "fraction_le_0_01": (
                sum(x <= 0.01 for x in probability_margins)
                / len(probability_margins)
            ),
            "fraction_le_0_05": (
                sum(x <= 0.05 for x in probability_margins)
                / len(probability_margins)
            ),
            "fraction_le_0_10": (
                sum(x <= 0.10 for x in probability_margins)
                / len(probability_margins)
            ),
        },
        "logit_margin": {
            "mean": mean(logit_margins),
            "median": median(logit_margins),
            "min": min(logit_margins),
            "p10": _quantile(logit_margins, 0.10),
            "p25": _quantile(logit_margins, 0.25),
        },
    }


def _load_model(
    *,
    model_name: str,
    revision: str,
    adapter_path: Path | None,
):
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
    if adapter_path is not None:
        model = PeftModel.from_pretrained(base, str(adapter_path))
    else:
        model = base
    model.eval()
    return model


def profile_trajectory_margins(
    *,
    tasks_path: Path,
    model_name: str,
    revision: str,
    adapter_path: Path | None,
    greedy_predictions_path: Path | None,
    output_path: Path,
    max_new_tokens: int = 512,
    primary_window: int = 128,
) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "trajectory margin analysis requires model dependencies"
        ) from exc

    tasks = list(load_tasks(tasks_path).values())
    stored = (
        {
            (str(row["task_id"]), int(row["sample_id"])): row
            for row in load_jsonl(greedy_predictions_path)
        }
        if greedy_predictions_path is not None
        else {}
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=True,
    )
    model = _load_model(
        model_name=model_name,
        revision=revision,
        adapter_path=adapter_path,
    )

    rows: dict[str, Any] = {}
    raw_match_count = 0

    with torch.inference_mode():
        for task in tasks:
            prompt = build_prompt(task.prompt, task.starter_code)
            encoded = tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded["attention_mask"].to(model.device)
            prompt_length = input_ids.shape[1]

            generated_full = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )[0]
            generated_ids = generated_full[prompt_length:]

            raw_completion = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )
            stored_row = stored.get((task.task_id, 0))
            raw_matches = (
                None
                if greedy_predictions_path is None
                else (
                    stored_row is not None
                    and str(stored_row.get("raw_completion", ""))
                    == raw_completion
                )
            )
            if raw_matches is not None:
                raw_match_count += int(raw_matches)

            token_ids = generated_ids.tolist()
            while token_ids and token_ids[-1] in {
                tokenizer.eos_token_id,
                tokenizer.pad_token_id,
            }:
                token_ids.pop()

            if not token_ids:
                rows[task.task_id] = {
                    "task_id": task.task_id,
                    "raw_matches_exp004f": raw_matches,
                    "completion_token_count": 0,
                    "generated_top1_agreement_fraction": None,
                    "first128": _margin_summary([], []),
                    "full": _margin_summary([], []),
                }
                continue

            full_ids = torch.tensor(
                [input_ids[0].tolist() + token_ids],
                device=model.device,
            )
            full_mask = torch.ones_like(full_ids)
            logits = model(
                input_ids=full_ids,
                attention_mask=full_mask,
                use_cache=False,
            ).logits[0]

            prediction_logits = logits[
                prompt_length - 1 : prompt_length - 1 + len(token_ids)
            ].float()
            top2 = torch.topk(prediction_logits, k=2, dim=-1)
            top2_logits = top2.values
            top1_ids = top2.indices[:, 0]
            log_norm = torch.logsumexp(prediction_logits, dim=-1)
            top2_prob = torch.exp(top2_logits - log_norm[:, None])

            probability_margins = (
                top2_prob[:, 0] - top2_prob[:, 1]
            ).cpu().tolist()
            logit_margins = (
                top2_logits[:, 0] - top2_logits[:, 1]
            ).cpu().tolist()

            generated_tensor = torch.tensor(
                token_ids,
                device=top1_ids.device,
            )
            top1_agreement = float(
                (top1_ids == generated_tensor).float().mean().item()
            )

            window = min(primary_window, len(token_ids))
            rows[task.task_id] = {
                "task_id": task.task_id,
                "raw_matches_exp004f": raw_matches,
                "completion_token_count": len(token_ids),
                "generated_top1_agreement_fraction": top1_agreement,
                "first128": _margin_summary(
                    probability_margins[:window],
                    logit_margins[:window],
                ),
                "full": _margin_summary(
                    probability_margins,
                    logit_margins,
                ),
            }

            del logits, prediction_logits, top2, top2_logits
            del top2_prob, log_norm, generated_full, generated_ids

    result = {
        "model": model_name,
        "revision": revision,
        "adapter": str(adapter_path) if adapter_path is not None else None,
        "tasks": len(tasks),
        "primary_window": primary_window,
        "raw_match_count_vs_reference": (
            raw_match_count if greedy_predictions_path is not None else None
        ),
        "raw_match_fraction_vs_reference": (
            raw_match_count / len(tasks)
            if greedy_predictions_path is not None
            else None
        ),
        "per_task": rows,
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
                "primary_window": primary_window,
                "raw_match_fraction_vs_reference": result[
                    "raw_match_fraction_vs_reference"
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
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--greedy-predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--primary-window", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile_trajectory_margins(
        tasks_path=args.tasks,
        model_name=args.model,
        revision=args.revision,
        adapter_path=args.adapter,
        greedy_predictions_path=args.greedy_predictions,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
        primary_window=args.primary_window,
    )


if __name__ == "__main__":
    main()
