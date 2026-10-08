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

from coderl_lab.generation import build_prompt


def load_sft_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            fmt = str(row.get("format", "code"))
            required = (
                ("task_id", "prompt", "starter_code", "response")
                if fmt == "code" else ("task_id", "prompt", "response")
            )
            if fmt not in {"code", "raw"}:
                raise ValueError(f"SFT line {line_no} has unsupported format: {fmt}")
            missing = [key for key in required if key not in row]
            if missing:
                raise ValueError(
                    f"SFT line {line_no} missing: {', '.join(missing)}"
                )
            rows.append(row)
    if not rows:
        raise ValueError("SFT dataset is empty")
    return rows


def prepare_prompt_completion_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    max_samples: int | None = None,
) -> list[dict[str, str]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    if max_samples is not None:
        shuffled = shuffled[:max_samples]

    formatted: list[dict[str, str]] = []
    for row in shuffled:
        sample_format = str(row.get("format", "code"))
        if sample_format == "raw":
            prompt = str(row["prompt"])
            if not prompt.strip():
                raise ValueError("raw SFT prompt cannot be empty")
        elif sample_format == "code":
            prompt = build_prompt(
                str(row["prompt"]),
                str(row.get("starter_code", "")),
            )
        else:
            raise ValueError(f"unsupported SFT prompt format: {sample_format}")
        formatted.append({
            "prompt": prompt,
            "completion": str(row["response"]).rstrip() + "\n",
        })
    return formatted


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("training config must be a mapping")
    return data


def compute_warmup_steps(
    *,
    num_examples: int,
    epochs: float,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    warmup_ratio: float,
) -> int:
    """Convert the configured warmup ratio into optimizer update steps.

    TRL 1.14.1 exposes warmup_steps rather than warmup_ratio. Keep the
    experiment configuration expressed as a ratio, but materialize an integer
    number of optimizer steps deterministically for the pinned runtime.
    """
    if num_examples <= 0:
        raise ValueError("num_examples must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if per_device_batch_size <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("batch sizes must be positive")
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError("warmup_ratio must be in [0, 1]")

    micro_batches_per_epoch = math.ceil(num_examples / per_device_batch_size)
    update_steps_per_epoch = math.ceil(
        micro_batches_per_epoch / gradient_accumulation_steps
    )
    total_update_steps = math.ceil(update_steps_per_epoch * epochs)

    if warmup_ratio == 0.0:
        return 0
    return min(
        total_update_steps,
        max(1, math.ceil(total_update_steps * warmup_ratio)),
    )


def run_sft(
    *,
    config_path: Path,
    data_path: Path,
    output_dir: Path | None = None,
    max_samples: int | None = None,
    num_train_epochs: float | None = None,
    max_eval_samples: int | None = None,
) -> dict[str, Any]:
    try:
        import peft
        import torch
        import transformers
        import trl
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "training dependencies are missing; install with "
            "pip install -e '.[model]'"
        ) from exc

    config = load_config(config_path)
    model_cfg = dict(config["model"])
    train_cfg = dict(config["training"])
    lora_cfg = dict(config["lora"])

    seed = int(train_cfg.get("seed", 42))
    rows = load_sft_rows(data_path)
    prepared_rows = prepare_prompt_completion_rows(
        rows,
        seed=seed,
        max_samples=max_samples,
    )
    dataset = Dataset.from_list(prepared_rows)

    # Held-out validation is optional; never concatenate it into the train set.
    eval_path_raw = config.get("data", {}).get("heldout_validation")
    evaluation_dataset = None
    if eval_path_raw is not None:
        evaluation_rows = load_sft_rows(Path(str(eval_path_raw)))
        if max_eval_samples is not None:
            if max_eval_samples <= 0:
                raise ValueError("max_eval_samples must be positive")
            evaluation_rows = evaluation_rows[:max_eval_samples]
        evaluation_prepared = prepare_prompt_completion_rows(
            evaluation_rows, seed=seed
        )
        train_task_ids = {str(row["task_id"]) for row in rows}
        evaluation_task_ids = {str(row["task_id"]) for row in evaluation_rows}
        if train_task_ids & evaluation_task_ids:
            raise ValueError("SFT train/held-out validation task_id overlap")
        train_prompts = {str(row["prompt"]) for row in prepared_rows}
        validation_prompts = {str(row["prompt"]) for row in evaluation_prepared}
        if train_prompts & validation_prompts:
            raise ValueError("SFT train/held-out validation prompt overlap")
        evaluation_dataset = Dataset.from_list(evaluation_prepared)

    model_name = str(model_cfg["name_or_path"])
    revision = str(model_cfg["revision"])
    final_output = output_dir or Path(str(train_cfg["output_dir"]))
    final_output.mkdir(parents=True, exist_ok=True)

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

    dtype_name = str(model_cfg.get("dtype", "bfloat16"))
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]

    # Track A: optional NF4 QLoRA; default behavior remains ordinary bf16 LoRA
    # so all established EXP-002/006A baselines remain unchanged.
    quant_cfg = dict(config.get("quantization", {}))
    quant_mode = str(quant_cfg.get("mode", "none"))
    if quant_mode not in {"none", "nf4"}:
        raise ValueError(f"unsupported quantization mode: {quant_mode}")
    model_kwargs: dict[str, Any] = {
        "revision": revision,
        "dtype": dtype,
        "trust_remote_code": True,
    }
    if quant_mode == "nf4":
        if not torch.cuda.is_available():
            raise RuntimeError("NF4 QLoRA requires a CUDA GPU")
        try:
            from peft import prepare_model_for_kbit_training
            from transformers import BitsAndBytesConfig
            import bitsandbytes
        except ImportError as exc:
            raise RuntimeError("NF4 QLoRA requires pip install bitsandbytes") from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=bool(quant_cfg.get("double_quant", True)),
            bnb_4bit_compute_dtype=dtype,
        )
        model_kwargs["device_map"] = {"": torch.cuda.current_device()}
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = False
    if quant_mode == "nf4":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        )

    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg.get("dropout", 0.0)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lora_cfg["target_modules"]),
    )

    epochs = (
        float(num_train_epochs)
        if num_train_epochs is not None
        else float(train_cfg["num_train_epochs"])
    )
    per_device_batch_size = int(train_cfg["per_device_train_batch_size"])
    gradient_accumulation_steps = int(train_cfg["gradient_accumulation_steps"])
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.0))
    warmup_steps = compute_warmup_steps(
        num_examples=len(dataset),
        epochs=epochs,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
    )

    args = SFTConfig(
        output_dir=str(final_output),
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=float(train_cfg["learning_rate"]),
        warmup_steps=warmup_steps,
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "cosine")),
        logging_steps=int(train_cfg.get("logging_steps", 1)),
        save_strategy=str(train_cfg.get("save_strategy", "no")),
        eval_strategy="epoch" if evaluation_dataset is not None else "no",
        seed=seed,
        data_seed=seed,
        bf16=dtype_name == "bfloat16",
        fp16=dtype_name == "float16",
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        max_length=int(train_cfg.get("max_length", 1024)),
        packing=bool(train_cfg.get("packing", False)),
        completion_only_loss=True,
        report_to="none",
        remove_unused_columns=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=evaluation_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    started = time.perf_counter()
    train_result = trainer.train()
    elapsed = time.perf_counter() - started

    trainer.save_model(str(final_output))
    tokenizer.save_pretrained(str(final_output))
    saved_adapter_path = final_output / "adapter_model.safetensors"
    if not saved_adapter_path.is_file():
        raise FileNotFoundError(
            f"SFT did not save expected LoRA adapter weights: {saved_adapter_path}"
        )
    adapter_sha256 = hashlib.sha256(saved_adapter_path.read_bytes()).hexdigest()
    train_data_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    heldout_data_sha256 = (
        hashlib.sha256(Path(str(eval_path_raw)).read_bytes()).hexdigest()
        if eval_path_raw is not None else None
    )

    trainable = sum(
        p.numel() for p in trainer.model.parameters() if p.requires_grad
    )
    total = sum(p.numel() for p in trainer.model.parameters())

    summary: dict[str, Any] = {
        "config_path": str(config_path),
        "data_path": str(data_path),
        "output_dir": str(final_output),
        "model": model_name,
        "requested_model_revision": revision,
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "dtype": dtype_name,
        "quantization": {
            "mode": quant_mode,
            "double_quant": bool(quant_cfg.get("double_quant", True))
            if quant_mode == "nf4" else None,
            "compute_dtype": dtype_name if quant_mode == "nf4" else None,
        },
        "num_examples": len(dataset),
        "saved_adapter_sha256": adapter_sha256,
        "train_data_sha256": train_data_sha256,
        "heldout_data_sha256": heldout_data_sha256,
        "num_heldout_validation_examples": (
            len(evaluation_dataset) if evaluation_dataset is not None else 0
        ),
        "num_train_epochs": epochs,
        "warmup_ratio_requested": warmup_ratio,
        "warmup_steps": warmup_steps,
        "seed": seed,
        "lora": {
            "r": peft_config.r,
            "alpha": peft_config.lora_alpha,
            "dropout": peft_config.lora_dropout,
            "target_modules": list(lora_cfg["target_modules"]),
            "trainable_parameters": trainable,
            "total_parameters": total,
            "trainable_fraction": trainable / total,
        },
        "metrics": dict(train_result.metrics),
        "elapsed_seconds_wall": elapsed,
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

    (final_output / "log_history.json").write_text(
        json.dumps(list(trainer.state.log_history), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (final_output / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CodeRL-Lab SFT")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--num-train-epochs", type=float)
    parser.add_argument("--max-eval-samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_sft(
        config_path=args.config,
        data_path=args.data,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        num_train_epochs=args.num_train_epochs,
        max_eval_samples=args.max_eval_samples,
    )


if __name__ == "__main__":
    main()
