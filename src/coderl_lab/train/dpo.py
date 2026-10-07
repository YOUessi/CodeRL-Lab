from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import yaml


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("DPO config must be a mapping")
    return data


def compute_warmup_steps(
    *,
    num_examples: int,
    epochs: float,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    warmup_ratio: float,
    max_steps: int,
) -> int:
    if max_steps > 0:
        total_steps = max_steps
    else:
        micro_batches = math.ceil(num_examples / per_device_batch_size)
        steps_per_epoch = math.ceil(
            micro_batches / gradient_accumulation_steps
        )
        total_steps = max(1, math.ceil(steps_per_epoch * epochs))
    if warmup_ratio <= 0:
        return 0
    return max(1, math.ceil(total_steps * warmup_ratio))


def override_lora_dropout(model, value: float) -> dict[str, Any]:
    """Override only PEFT LoRA dropout modules.

    This is used for EXP-004C causal controls. It leaves all other model
    dropout untouched and records the previous probabilities.
    """
    if not 0.0 <= value <= 1.0:
        raise ValueError("LoRA dropout override must be in [0, 1]")
    changed: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if "lora_dropout" not in name:
            continue
        if not hasattr(module, "p"):
            continue
        old = float(module.p)
        module.p = float(value)
        changed.append({"name": name, "old": old, "new": float(value)})
    return {
        "requested": float(value),
        "modules_changed": len(changed),
        "previous_values": sorted({row["old"] for row in changed}),
        "modules": changed,
    }


def prepare_rows(
    path: Path,
    *,
    seed: int,
    max_pairs: int | None = None,
) -> list[dict[str, str]]:
    raw = load_jsonl(path)
    rows = [
        {
            "prompt": str(x["prompt"]),
            "chosen": str(x["chosen"]),
            "rejected": str(x["rejected"]),
        }
        for x in raw
    ]
    random.Random(seed).shuffle(rows)
    if max_pairs is not None:
        rows = rows[:max_pairs]
    if not rows:
        raise ValueError("DPO preference dataset is empty")
    return rows


def run_dpo(
    *,
    config_path: Path,
    preferences_path: Path,
    sft_adapter_path: Path,
    output_dir: Path | None = None,
    max_pairs: int | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    try:
        import peft
        import torch
        import transformers
        import trl
        from datasets import Dataset
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "DPO dependencies are missing; install with "
            "pip install -e '.[model]'"
        ) from exc

    cfg = load_config(config_path)
    model_cfg = dict(cfg["model"])
    initial_cfg = dict(cfg["initial_policy"])
    train_cfg = dict(cfg["training"])

    seed = int(train_cfg.get("seed", 42))
    rows = prepare_rows(
        preferences_path,
        seed=seed,
        max_pairs=max_pairs,
    )
    dataset = Dataset.from_list(rows)

    model_name = str(model_cfg["name_or_path"])
    revision = str(model_cfg["revision"])
    dtype_name = str(model_cfg.get("dtype", "bfloat16"))
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]

    final_output = output_dir or Path(str(train_cfg["output_dir"]))
    final_output.mkdir(parents=True, exist_ok=True)

    adapter_weights = sft_adapter_path / "adapter_model.safetensors"
    if not adapter_weights.exists():
        raise FileNotFoundError(f"SFT adapter missing: {adapter_weights}")
    adapter_sha = hashlib.sha256(adapter_weights.read_bytes()).hexdigest()
    expected_sha = str(initial_cfg["expected_sha256"])
    if adapter_sha != expected_sha:
        raise ValueError(
            f"SFT adapter SHA mismatch: expected {expected_sha}, got {adapter_sha}"
        )

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        dtype=dtype,
        trust_remote_code=True,
    )
    base_model.config.use_cache = False

    model = PeftModel.from_pretrained(
        base_model,
        str(sft_adapter_path),
        is_trainable=True,
    )

    adapter_dropout_override = train_cfg.get("adapter_dropout_override")
    dropout_override_summary = None
    if adapter_dropout_override is not None:
        dropout_override_summary = override_lora_dropout(
            model,
            float(adapter_dropout_override),
        )

    epochs = float(train_cfg.get("num_train_epochs", 3.0))
    requested_max_steps = (
        int(max_steps)
        if max_steps is not None
        else int(train_cfg.get("max_steps", -1))
    )
    batch_size = int(train_cfg["per_device_train_batch_size"])
    grad_accum = int(train_cfg["gradient_accumulation_steps"])
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.0))
    warmup_steps = compute_warmup_steps(
        num_examples=len(dataset),
        epochs=epochs,
        per_device_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_ratio=warmup_ratio,
        max_steps=requested_max_steps,
    )

    args = DPOConfig(
        output_dir=str(final_output),
        num_train_epochs=epochs,
        max_steps=requested_max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(train_cfg["learning_rate"]),
        warmup_steps=warmup_steps,
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "cosine")),
        beta=float(train_cfg.get("beta", 0.1)),
        loss_type=str(train_cfg.get("loss_type", "sigmoid")),
        max_length=int(train_cfg.get("max_length", 1024)),
        logging_steps=int(train_cfg.get("logging_steps", 1)),
        save_strategy=str(train_cfg.get("save_strategy", "no")),
        seed=seed,
        data_seed=seed,
        bf16=dtype_name == "bfloat16",
        fp16=dtype_name == "float16",
        gradient_checkpointing=bool(
            train_cfg.get("gradient_checkpointing", True)
        ),
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    started = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - started

    trainer.save_model(str(final_output))
    tokenizer.save_pretrained(str(final_output))

    history = list(trainer.state.log_history)
    (final_output / "log_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    saved_adapter = final_output / "adapter_model.safetensors"
    output_sha = (
        hashlib.sha256(saved_adapter.read_bytes()).hexdigest()
        if saved_adapter.exists()
        else None
    )

    trainable = sum(
        p.numel() for p in trainer.model.parameters() if p.requires_grad
    )
    total = sum(p.numel() for p in trainer.model.parameters())

    summary: dict[str, Any] = {
        "config_path": str(config_path),
        "preferences_path": str(preferences_path),
        "sft_adapter_path": str(sft_adapter_path),
        "sft_adapter_sha256": adapter_sha,
        "output_dir": str(final_output),
        "model": model_name,
        "requested_model_revision": revision,
        "num_pairs": len(dataset),
        "num_train_epochs": epochs,
        "max_steps": requested_max_steps,
        "warmup_ratio_requested": warmup_ratio,
        "warmup_steps": warmup_steps,
        "seed": seed,
        "beta": args.beta,
        "loss_type": args.loss_type,
        "adapter_dropout_override": dropout_override_summary,
        "global_step": trainer.state.global_step,
        "trainer_max_steps": trainer.state.max_steps,
        "final_epoch": trainer.state.epoch,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "metrics": dict(result.metrics),
        "elapsed_seconds_wall": elapsed,
        "output_adapter_sha256": output_sha,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
            "peft": peft.__version__,
        },
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        summary["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }

    (final_output / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run EXP-004B repair DPO")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--preferences", type=Path, required=True)
    p.add_argument("--sft-adapter", type=Path, required=True)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--max-pairs", type=int)
    p.add_argument("--max-steps", type=int)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dpo(
        config_path=args.config,
        preferences_path=args.preferences,
        sft_adapter_path=args.sft_adapter,
        output_dir=args.output_dir,
        max_pairs=args.max_pairs,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
