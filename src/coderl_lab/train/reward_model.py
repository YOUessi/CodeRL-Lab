"""Pairwise reward-model training using a frozen UltraFeedback snapshot.

TRAIN-008A is a reward-modeling prerequisite for RLHF/PPO, not PPO itself.
Data validation and metric helpers are CPU-only; CUDA/model imports are lazy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path
from statistics import mean
from typing import Any

import yaml


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_pairs(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            prompt = str(raw.get("prompt", ""))
            chosen = str(raw.get("chosen", ""))
            rejected = str(raw.get("rejected", ""))
            if not prompt.strip() or not chosen.strip() or not rejected.strip():
                raise ValueError(f"missing prompt/candidate in pair {lineno}")
            if chosen.strip() == rejected.strip():
                raise ValueError(f"equal chosen and rejected in pair {lineno}")
            if prompt in seen:
                raise ValueError(f"duplicate preference prompt in pair {lineno}")
            seen.add(prompt)
            if not prompt.rstrip().endswith("### Assistant:"):
                raise ValueError(f"unexpected prompt formatting in pair {lineno}")
            rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    if not rows:
        raise ValueError("empty reward model preference dataset")
    return rows


def render_pair(row: dict[str, str]) -> tuple[str, str]:
    """Identical prompt prefix in both arms; only completion content changes."""
    return row["prompt"] + row["chosen"], row["prompt"] + row["rejected"]


def pack_reward_tokens(
    tokenizer,
    *,
    prompt: str,
    completion: str,
    max_length: int,
    max_prompt_tokens: int,
    min_response_tokens: int,
) -> tuple[list[int], dict[str, Any]]:
    """Preserve the assistant response even for long conversation prefixes.

    Naively doing tokenizer(prompt + completion, truncation=True) can truncate
    both preferred/rejected responses away, making their reward scores identical.
    Keep the same tail of the prompt for both answers, reserve response room,
    retain the *suffix* of a very long response, and use the model's EOS token.
    This is a frozen representation policy, not an outcome-dependent truncation.
    """
    if not prompt.strip() or not completion.strip():
        raise ValueError("reward prompt/completion cannot be empty")
    if max_length < 8 or min_response_tokens < 1 or max_prompt_tokens < 1:
        raise ValueError("invalid reward token budget")
    if max_prompt_tokens + min_response_tokens + 1 > max_length:
        raise ValueError("prompt plus reserved answer tokens exceed max_length")
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("reward tokenizer must define EOS token")
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    completion_ids = list(tokenizer.encode(completion, add_special_tokens=False))
    if not completion_ids:
        raise ValueError("tokenized reward completion cannot be empty")
    shared_prompt = prompt_ids[-max_prompt_tokens:]
    answer_budget = max_length - len(shared_prompt) - 1
    if answer_budget < min_response_tokens:
        raise AssertionError("reserved answer budget not met")
    kept_response = completion_ids[-answer_budget:]
    packed = shared_prompt + kept_response + [int(eos)]
    if len(packed) > max_length or not kept_response:
        raise AssertionError("invalid reward packed sequence")
    return packed, {
        "prompt_original_tokens": len(prompt_ids),
        "prompt_used_tokens": len(shared_prompt),
        "answer_original_tokens": len(completion_ids),
        "answer_used_tokens": len(kept_response),
        "prompt_truncated": len(shared_prompt) < len(prompt_ids),
        "answer_truncated": len(kept_response) < len(completion_ids),
    }


def pairwise_logistic_loss(margin: float) -> float:
    """Numerically stable -log(sigmoid(r_chosen - r_rejected))."""
    if margin >= 0:
        return math.log1p(math.exp(-margin))
    return -margin + math.log1p(math.exp(margin))


def compute_reward_metrics(margins: list[float]) -> dict[str, float | int]:
    if not margins or not all(math.isfinite(x) for x in margins):
        raise ValueError("reward margins must be non-empty finite numbers")
    return {
        "pairs": len(margins),
        "preference_accuracy": sum(x > 0 for x in margins) / len(margins),
        "ties": sum(x == 0 for x in margins),
        "mean_margin": mean(margins),
        "mean_pairwise_loss": mean(pairwise_logistic_loss(x) for x in margins),
    }


def validate_preference_snapshot(
    *,
    config_path: Path,
    train_path: Path,
    heldout_path: Path,
    manifest_path: Path,
    train_limit: int | None = None,
    eval_limit: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("experiment", {}).get("id") != "TRAIN-008A":
        raise ValueError("wrong TRAIN-008A configuration")
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = cfg["data"]
    if any((
        m.get("track") != "A-posttraining",
        m.get("stage") != "dpo",
        m.get("full_scan") is not True,
        m.get("prompt_overlap") != 0,
        m.get("dataset_revision") != data["revision"],
        m["train"]["rows"] != data["expected_train_pairs"],
        m["validation"]["rows"] != data["expected_heldout_pairs"],
        file_sha256(train_path) != m["train"]["sha256"],
        file_sha256(heldout_path) != m["validation"]["sha256"],
    )):
        raise ValueError("frozen full preference manifest or hashes mismatch")

    all_train = read_pairs(train_path)
    all_eval = read_pairs(heldout_path)
    if len(all_train) != m["train"]["rows"] or len(all_eval) != m["validation"]["rows"]:
        raise ValueError("preference row count mismatch")
    if {x["prompt"] for x in all_train} & {x["prompt"] for x in all_eval}:
        raise ValueError("train/heldout preference prompt leakage")
    if train_limit is not None and (train_limit < 1 or train_limit > len(all_train)):
        raise ValueError("bad train smoke limit")
    if eval_limit is not None and (eval_limit < 1 or eval_limit > len(all_eval)):
        raise ValueError("bad validation smoke limit")
    train = all_train[:train_limit] if train_limit is not None else all_train
    heldout = all_eval[:eval_limit] if eval_limit is not None else all_eval
    identity = {
        "experiment": "TRAIN-008A",
        "config_sha256": file_sha256(config_path),
        "manifest_sha256": file_sha256(manifest_path),
        "train_file_sha256": file_sha256(train_path),
        "heldout_file_sha256": file_sha256(heldout_path),
        "train_rows": len(train), "heldout_rows": len(heldout),
        "seed": cfg["training"]["seed"],
        "model_revision": cfg["model"]["revision"],
        "formal": train_limit is None and eval_limit is None,
    }
    return cfg, train, heldout, identity


def _checkpoint_path(root: Path, step: int) -> Path:
    return root / f"checkpoint-{step}"


def validate_resume(
    *, output_dir: Path, checkpoint: Path, identity: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    resolved = checkpoint.resolve()
    if resolved.parent != output_dir.resolve():
        raise ValueError("reward checkpoint outside run directory")
    if not resolved.name.startswith("checkpoint-"):
        raise ValueError("reward checkpoint name invalid")
    filename = resolved / "state.pt"
    model_weights = resolved / "adapter_model.safetensors"
    if not filename.is_file() or not model_weights.is_file():
        raise FileNotFoundError("incomplete reward-model checkpoint")
    frozen_path = output_dir / "training_identity.json"
    if not frozen_path.exists():
        raise FileNotFoundError("reward-model identity absent")
    if json.loads(frozen_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("reward-model config/data identity mismatch")
    if json.loads((resolved / "checkpoint_identity.json").read_text(encoding="utf-8")) != identity:
        raise ValueError("reward checkpoint identity mismatch")
    return resolved, {"path": str(resolved), "state_file": str(filename)}


def _save_checkpoint(
    *,
    root: Path,
    step: int,
    identity: dict[str, Any],
    model,
    optimizer,
    scheduler,
    processed_pairs: int,
    cumulative_loss: float,
    limit: int,
) -> None:
    import torch
    dest = _checkpoint_path(root, step)
    if dest.exists():
        raise FileExistsError(f"checkpoint exists; refusing overwrite: {dest}")
    temp = root / f".checkpoint-{step}-unfinished"
    if temp.exists():
        raise FileExistsError(f"temporary checkpoint exists: {temp}")
    temp.mkdir(parents=True)
    try:
        model.save_pretrained(temp, safe_serialization=True)
        torch.save(
            {
                "step": step,
                "processed_pairs": processed_pairs,
                "cumulative_loss": cumulative_loss,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
            },
            temp / "state.pt",
        )
        (temp / "checkpoint_identity.json").write_text(
            json.dumps(identity, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temp.rename(dest)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    checkpoints = sorted(
        (p for p in root.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    for previous in checkpoints[:-limit]:
        shutil.rmtree(previous)


def train_reward_model(
    *,
    config_path: Path,
    output_dir: Path | None = None,
    train_limit: int | None = None,
    eval_limit: int | None = None,
    resume_from_checkpoint: Path | None = None,
) -> dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as F
        from transformers import (
            AutoModelForSequenceClassification, AutoTokenizer,
            BitsAndBytesConfig, get_cosine_schedule_with_warmup,
        )
        from peft import (
            LoraConfig, TaskType, PeftModel, get_peft_model,
            prepare_model_for_kbit_training,
        )
    except ImportError as exc:
        raise RuntimeError("Reward modeling requires installed torch/transformers/peft/bitsandbytes") from exc

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = output_dir or Path(str(cfg["training"]["output_dir"]))
    data_cfg = cfg["data"]
    cfg, train, heldout, identity = validate_preference_snapshot(
        config_path=config_path,
        train_path=Path(data_cfg["train"]),
        heldout_path=Path(data_cfg["heldout"]),
        manifest_path=Path(data_cfg["manifest"]),
        train_limit=train_limit, eval_limit=eval_limit,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("TRAIN-008A requires a real CUDA GPU")
    if cfg["quantization"]["mode"] != "nf4":
        raise ValueError("first reward baseline is frozen to NF4")
    root.mkdir(parents=True, exist_ok=True)
    identity_file = root / "training_identity.json"
    if resume_from_checkpoint is None:
        if identity_file.exists() or any(root.glob("checkpoint-*")) or (root / "adapter_model.safetensors").exists():
            raise FileExistsError("existing reward model state; explicit resume required")
        identity_file.write_text(json.dumps(identity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        validate_resume(output_dir=root, checkpoint=resume_from_checkpoint, identity=identity)

    setup = cfg["training"]
    seed = int(setup["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()

    revision = str(cfg["model"]["revision"])
    model_name = str(cfg["model"]["name_or_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    from torch import bfloat16
    base = AutoModelForSequenceClassification.from_pretrained(
        model_name, revision=revision, num_labels=1,
        dtype=bfloat16, trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=bool(cfg["quantization"]["double_quant"]),
            bnb_4bit_compute_dtype=bfloat16,
        ),
        device_map={"": torch.cuda.current_device()},
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(
        base, use_gradient_checkpointing=bool(setup["gradient_checkpointing"]),
    )
    lora = cfg["lora"]
    if resume_from_checkpoint is None:
        model = get_peft_model(
            base,
            LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=int(lora["r"]), lora_alpha=int(lora["alpha"]),
                lora_dropout=float(lora["dropout"]),
                target_modules=list(lora["target_modules"]),
                modules_to_save=list(lora["modules_to_save"]),
            ),
        )
    else:
        model = PeftModel.from_pretrained(
            base, str(resume_from_checkpoint), is_trainable=True,
        )
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise RuntimeError("reward model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters, lr=float(setup["learning_rate"]),
        weight_decay=float(setup.get("weight_decay", 0.0)),
    )
    gradient_accum = int(setup["gradient_accumulation_steps"])
    if gradient_accum < 1:
        raise ValueError("invalid gradient accumulation")
    if int(setup["epochs"]) != 1:
        raise ValueError("TRAIN-008A first formal protocol supports exactly one epoch")
    total_steps = math.ceil(len(train) / gradient_accum)
    warmup = math.ceil(total_steps * float(setup["warmup_ratio"]))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    ids = list(range(len(train)))
    random.Random(seed).shuffle(ids)
    total_loss = 0.0
    processed = 0
    step = 0
    if resume_from_checkpoint is not None:
        # This local checkpoint is produced solely by our own save routine.
        state = torch.load(
            resume_from_checkpoint / "state.pt", map_location="cpu",
            weights_only=False,
        )
        step = int(state["step"])
        processed = int(state["processed_pairs"])
        if not (0 <= processed <= len(train) and step == processed // gradient_accum):
            raise ValueError("reward-model checkpoint step/data progress mismatch")
        total_loss = float(state["cumulative_loss"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])

    max_length = int(setup["max_length"])
    max_prompt_tokens = int(setup.get("max_prompt_tokens", 512))
    min_response_tokens = int(setup.get("min_response_tokens", 128))
    if max_prompt_tokens + min_response_tokens + 1 > max_length:
        raise ValueError("reward packed token budget is invalid")
    log: list[dict[str, Any]] = []
    started = time.perf_counter()

    def reward_score(prompt: str, completion: str):
        token_ids, _ = pack_reward_tokens(
            tokenizer,
            prompt=prompt, completion=completion,
            max_length=max_length,
            max_prompt_tokens=max_prompt_tokens,
            min_response_tokens=min_response_tokens,
        )
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=model.device)
        attention_mask = torch.ones_like(input_ids)
        return model(input_ids=input_ids, attention_mask=attention_mask).logits.float().reshape(-1)[0]

    baseline_eval_path = root / "initial_heldout_metrics.json"
    if resume_from_checkpoint is None:
        if baseline_eval_path.exists():
            raise FileExistsError("initial reward evaluation already exists")
        model.eval()
        with torch.inference_mode():
            initial_margins = [
                float((reward_score(row["prompt"], row["chosen"]) -
                       reward_score(row["prompt"], row["rejected"])).item())
                for row in heldout
            ]
        baseline_heldout = compute_reward_metrics(initial_margins)
        baseline_eval_path.write_text(
            json.dumps(baseline_heldout, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        if not baseline_eval_path.is_file():
            raise FileNotFoundError("frozen initial reward ranking metrics missing on resume")
        baseline_heldout = json.loads(baseline_eval_path.read_text(encoding="utf-8"))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for pos in range(processed, len(train)):
        row = train[ids[pos]]
        chosen_r = reward_score(row["prompt"], row["chosen"])
        rejected_r = reward_score(row["prompt"], row["rejected"])
        margin = chosen_r - rejected_r
        loss = F.softplus(-margin)
        regularizer = float(setup.get("reward_l2", 0.0))
        loss = loss + regularizer * 0.5 * (chosen_r.square() + rejected_r.square())
        if not torch.isfinite(loss.detach()).item():
            raise FloatingPointError(f"reward loss became nonfinite at sample {pos}")
        (loss / gradient_accum).backward()
        total_loss += float(loss.detach().item())
        if (pos + 1) % gradient_accum != 0 and pos != len(train) - 1:
            continue
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float(setup["max_grad_norm"]))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        log.append({
            "step": step, "loss": float(loss.detach().item()),
            "margin": float(margin.detach().item()),
            "grad_norm": float(grad_norm),
            "processed_pairs": pos + 1,
            "lr": float(scheduler.get_last_lr()[0]),
        })
        if step % int(setup["save_steps"]) == 0:
            _save_checkpoint(
                root=root, step=step, identity=identity,
                model=model, optimizer=optimizer, scheduler=scheduler,
                processed_pairs=pos + 1, cumulative_loss=total_loss,
                limit=int(setup["save_total_limit"]),
            )
        if step % 8 == 0:
            print(f"TRAIN-008A pairwise optimizer step {step}/{total_steps}", flush=True)

    model.eval()
    eval_margins = []
    with torch.inference_mode():
        for row in heldout:
            margin = (
                reward_score(row["prompt"], row["chosen"]) -
                reward_score(row["prompt"], row["rejected"])
            ).item()
            eval_margins.append(float(margin))
    eval_metrics = compute_reward_metrics(eval_margins)
    elapsed = time.perf_counter() - started

    model.save_pretrained(root, safe_serialization=True)
    tokenizer.save_pretrained(root)
    adapter_file = root / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise FileNotFoundError("reward model adapter weights were not saved")
    result = {
        "experiment": "TRAIN-008A",
        "status": "formal" if identity["formal"] else "smoke_only",
        "initialization": "Qwen3-1.7B-Base fresh sequence-classification head, NF4/LoRA",
        "representation": {
            "max_length": max_length,
            "max_prompt_tokens": max_prompt_tokens,
            "min_response_tokens": min_response_tokens,
            "completion_preserved_for_every_pair": True,
            "long_prompt_policy": "left-truncate prompt to shared tail",
            "long_completion_policy": "keep completion suffix then EOS",
        },
        "training_identity": identity,
        "adapter_sha256": file_sha256(adapter_file),
        "train_pairs": len(train), "heldout_pairs": len(heldout),
        "optimizer_steps": step, "total_steps_expected": total_steps,
        "mean_training_loss": total_loss / len(train),
        "initial_heldout": baseline_heldout,
        "heldout": eval_metrics,
        "heldout_preference_accuracy_delta": (
            eval_metrics["preference_accuracy"] - baseline_heldout["preference_accuracy"]
        ),
        "resumed_from": str(resume_from_checkpoint) if resume_from_checkpoint is not None else None,
        "elapsed_seconds_wall": elapsed,
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "claim_limit": "preference ranking on frozen UltraFeedback heldout only; not proof of RLHF/PPO gains",
    }
    (root / "run_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (root / "log_history.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="TRAIN-008A frozen NF4 pairwise reward model")
    p.add_argument("--config", type=Path, default=Path("configs/posttrain_a_reward_model_qwen3_1.7b.yaml"))
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--smoke-train-pairs", type=int)
    p.add_argument("--smoke-eval-pairs", type=int)
    p.add_argument("--resume-from-checkpoint", type=Path)
    a = p.parse_args()
    if (a.smoke_train_pairs is None) != (a.smoke_eval_pairs is None):
        raise ValueError("smoke train/eval limits must be supplied together")
    train_reward_model(
        config_path=a.config, output_dir=a.output_dir,
        train_limit=a.smoke_train_pairs,
        eval_limit=a.smoke_eval_pairs,
        resume_from_checkpoint=a.resume_from_checkpoint,
    )


if __name__ == "__main__":
    main()
