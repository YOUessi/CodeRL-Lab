from __future__ import annotations

import argparse
import json
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
            required = ("task_id", "prompt", "starter_code", "response")
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

    return [
        {
            "prompt": build_prompt(
                str(row["prompt"]),
                str(row["starter_code"]),
            ),
            "completion": str(row["response"]).rstrip() + "\n",
        }
        for row in shuffled
    ]


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("training config must be a mapping")
    return data


def run_sft(
    *,
    config_path: Path,
    data_path: Path,
    output_dir: Path | None = None,
    max_samples: int | None = None,
    num_train_epochs: float | None = None,
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

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        dtype=dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False

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

    args = SFTConfig(
        output_dir=str(final_output),
        num_train_epochs=epochs,
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.0)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "cosine")),
        logging_steps=int(train_cfg.get("logging_steps", 1)),
        save_strategy=str(train_cfg.get("save_strategy", "no")),
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
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    started = time.perf_counter()
    train_result = trainer.train()
    elapsed = time.perf_counter() - started

    trainer.save_model(str(final_output))
    tokenizer.save_pretrained(str(final_output))

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
        "num_examples": len(dataset),
        "num_train_epochs": epochs,
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_sft(
        config_path=args.config,
        data_path=args.data,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        num_train_epochs=args.num_train_epochs,
    )


if __name__ == "__main__":
    main()
