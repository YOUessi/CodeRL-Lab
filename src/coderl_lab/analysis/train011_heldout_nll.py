"""TRAIN-011: frozen held-out completion-token NLL, base vs trained 7B adapter.

Both conditions use the *same* frozen UltraChat validation samples,
tokenizer, maximum length and quantized backbone. Training is not repeated.
This diagnostic is completion-token NLL, not a human preference judgement.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import random
import time
from pathlib import Path


MODEL = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
DATA_SHA = "a7f6e07f8569157fe5c3ae0deb65875c3427f9db3d6a5120379e1d9b10dae170"
ADAPTER_SHA = "25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(rows) != 256:
        raise ValueError(f"expected exactly 256 frozen heldout examples, got {len(rows)}")
    if any(r.get("format") != "raw" or not r.get("prompt") or not r.get("response") for r in rows):
        raise ValueError("invalid UltraChat prompt/completion row")
    ids = [str(x["task_id"]) for x in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate heldout task_id")
    return rows


def select_completion_positions(offsets: list[list[int]], prefix_chars: int) -> list[int]:
    """Return positions for tokens overlapping completion. Tokens at pos0 cannot be scored."""
    return [i for i, (begin, end) in enumerate(offsets)
            if i > 0 and end > prefix_chars and end > begin]


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("empty checkpoint")
    all_rows = list(rows)
    rows = [x for x in all_rows if int(x["target_tokens"]) > 0]
    if not rows:
        raise ValueError("no heldout samples have any completion tokens at frozen length 512")
    total = sum(int(x["target_tokens"]) for x in rows)
    if total <= 0:
        raise ValueError("no completion tokens")
    result = {
        "total_heldout_examples": len(all_rows),
        "evaluable_examples": len(rows),
        "unscorable_due_to_prompt_length": len(all_rows) - len(rows),
        "examples": len(rows),
        "target_tokens": total,
        "coverage_fraction": len(rows) / len(all_rows),
    }
    for name in ("base", "sft"):
        nll_sum = sum(float(x[f"{name}_sum_nll"]) for x in rows)
        mean_nll = nll_sum / total
        result[name] = {
            "total_nll": nll_sum,
            "token_weighted_mean_nll": mean_nll,
            "perplexity": math.exp(min(mean_nll, 70)),
            "mean_sample_nll": sum(float(x[f"{name}_sum_nll"]) / int(x["target_tokens"]) for x in rows) / len(rows),
        }
    result["sft_minus_base_token_mean_nll"] = (
        result["sft"]["token_weighted_mean_nll"] -
        result["base"]["token_weighted_mean_nll"]
    )
    result["examples_improved_nll"] = sum(x["sft_sum_nll"] < x["base_sum_nll"] for x in rows)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--heldout", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-examples", type=int, default=256)
    args = p.parse_args()

    if not 1 <= args.max_examples <= 256:
        raise ValueError("max-examples must be 1..256")
    if sha256(args.heldout) != DATA_SHA:
        raise ValueError("frozen UltraChat heldout data SHA mismatch")
    if sha256(args.adapter / "adapter_model.safetensors") != ADAPTER_SHA:
        raise ValueError("TRAIN-011 adapter SHA mismatch")
    rows = read_rows(args.heldout)

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True, use_fast=True)
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    backbone = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True, dtype=torch.bfloat16,
        quantization_config=quant, device_map={"": 0},
    )
    model = PeftModel.from_pretrained(backbone, str(args.adapter), is_trainable=False)
    model.eval()
    model.config.use_cache = False

    args.out.mkdir(parents=True, exist_ok=True)
    identity = {
        "experiment": "TRAIN-011-HELDOUT-NLL",
        "model": MODEL, "revision": REVISION,
        "adapter_sha256": ADAPTER_SHA, "heldout_sha256": DATA_SHA,
        "max_length": 512, "seed": 42,
        "scoring": "completion-only token NLL after concatenated prompt/completion tokenization; offsets select overlapping tokens",
        "precision": "NF4 double quant + BF16 compute",
        "no_completion_policy": "predefined length-based no-target samples recorded but not included in token NLL; never filter based on accuracy",
    }
    identity_path = args.out / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("existing checkpoint has different frozen identity")
    else:
        identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    checkpoint = args.out / "checkpoint.jsonl"
    saved = {}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if line.strip():
                x = json.loads(line)
                if x["index"] in saved:
                    raise ValueError("duplicate checkpoint index")
                saved[x["index"]] = x

    started = time.monotonic()
    with torch.inference_mode():
        for i, row in enumerate(rows[:args.max_examples]):
            if i in saved:
                continue
            prefix = str(row["prompt"])
            completion = str(row["response"]).rstrip() + "\n"
            enc = tokenizer(
                prefix + completion, return_offsets_mapping=True, truncation=True,
                max_length=512, return_tensors="pt",
            )
            offsets = enc.pop("offset_mapping")[0].tolist()
            selected = select_completion_positions(offsets, len(prefix))
            if not selected:
                # The training setup also truncates at 512. Such an input has no
                # assistant-token loss under the fixed protocol. Record every
                # index, including non-evaluable examples, without outcome selection.
                item = {
                    "index": i,
                    "task_id_sha256": hashlib.sha256(str(row["task_id"]).encode()).hexdigest(),
                    "target_tokens": 0,
                    "truncated_to_max_length": True,
                    "reason": "no_completion_token_within_train_max_length_512",
                }
                with checkpoint.open("a", encoding="utf-8") as h:
                    h.write(json.dumps(item, ensure_ascii=False) + "\n")
                saved[i] = item
                continue
            ids = enc["input_ids"].to(model.device)
            mask = enc["attention_mask"].to(model.device)
            target = ids[:, 1:]
            token_loss = {}
            for arm in ("base", "sft"):
                disable = model.disable_adapter() if arm == "base" else contextlib.nullcontext()
                with disable:
                    logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1, :]
                    # Compute only at the frozen target indices, avoiding a huge whole-vocab
                    # fp32 tensor for every prompt token.
                    picked = logits[:, [pos - 1 for pos in selected], :].float()
                    labels = target[:, [pos - 1 for pos in selected]]
                    value = F.cross_entropy(picked.transpose(1, 2), labels, reduction="sum")
                    token_loss[arm] = float(value.item())
                    del logits, picked
            item = {
                "index": i,
                "task_id_sha256": hashlib.sha256(str(row["task_id"]).encode()).hexdigest(),
                "target_tokens": len(selected),
                "truncated_to_max_length": bool(len(offsets) == 512),
                "base_sum_nll": token_loss["base"],
                "sft_sum_nll": token_loss["sft"],
            }
            with checkpoint.open("a", encoding="utf-8") as h:
                h.write(json.dumps(item, ensure_ascii=False) + "\n")
            saved[i] = item
            if (i + 1) % 16 == 0 or i + 1 == args.max_examples:
                print(f"TRAIN-011 heldout paired NLL {i+1}/{args.max_examples}", flush=True)

    if set(saved) != set(range(args.max_examples)):
        raise RuntimeError("incomplete contiguous evaluation checkpoint")
    summary = {
        **identity, "status": "completed",
        "requested_examples": args.max_examples,
        "formal_full_256": args.max_examples == 256,
        "evaluated_protocol_version": "length_512_v2_all_input_accounted",
        "metrics": aggregate([saved[i] for i in range(args.max_examples)]),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "cuda_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        "private_test_access": False,
        "limitations": "Token-level likelihood on frozen UltraChat answers, not actual human-judged generation quality. Quantized BF16 forward for both arms.",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
