from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
import random
import time
from pathlib import Path
from typing import Any

import yaml

from coderl_lab.execution import PythonExecutor
from coderl_lab.generation import build_prompt, extract_python_code
from coderl_lab.reward import compute_training_reward
from coderl_lab.schema import TestCase


def load_grpo_rows(
    tasks_path: Path,
    *,
    seed: int,
    max_tasks: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with tasks_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)

            required = (
                "task_id",
                "prompt",
                "entry_point",
                "starter_code",
                "public_tests",
            )
            missing = [key for key in required if key not in task]
            if missing:
                raise ValueError(
                    f"task line {line_no} missing fields: {', '.join(missing)}"
                )

            # Deliberately do not include hidden_tests in the trainer dataset.
            rows.append(
                {
                    "prompt": build_prompt(
                        str(task["prompt"]),
                        str(task.get("starter_code", "")),
                    ),
                    "task_id": str(task["task_id"]),
                    "entry_point": str(task["entry_point"]),
                    "public_tests": list(task["public_tests"]),
                    "setup_code": str(task.get("setup_code", "")),
                }
            )

    if not rows:
        raise ValueError("GRPO task dataset is empty")

    random.Random(seed).shuffle(rows)
    if max_tasks is not None:
        rows = rows[:max_tasks]
    return rows


def compute_warmup_steps(
    *,
    num_examples: int,
    num_generations: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    max_steps: int,
    num_train_epochs: float,
    warmup_ratio: float,
) -> int:
    """Materialize warmup ratio for the pinned TRL runtime."""
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError("warmup_ratio must be in [0, 1]")

    if per_device_train_batch_size <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("batch sizes must be positive")

    generation_batch_size = (
        per_device_train_batch_size * gradient_accumulation_steps
    )
    if generation_batch_size % num_generations != 0:
        raise ValueError(
            "per-device batch × gradient accumulation must be divisible "
            "by num_generations"
        )

    if max_steps > 0:
        total_steps = max_steps
    else:
        unique_prompts_per_update = generation_batch_size // num_generations
        optimizer_steps_per_epoch = math.ceil(
            num_examples / unique_prompts_per_update
        )
        total_steps = max(
            1,
            math.ceil(optimizer_steps_per_epoch * num_train_epochs),
        )

    if warmup_ratio == 0:
        return 0
    return max(1, math.ceil(total_steps * warmup_ratio))


class VerifiableCodeReward:
    """Public-test-only reward used by GRPO.

    Hidden tests are structurally absent from the dataset passed to this reward.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float,
        memory_limit: str,
        docker_image: str,
        max_workers: int = 1,
    ) -> None:
        self.executor = PythonExecutor(
            mode="docker",
            timeout_seconds=timeout_seconds,
            docker_image=docker_image,
            memory_limit=memory_limit,
        )
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self.max_workers = max_workers

    def _score_one(
        self,
        raw: str,
        entry_point: str,
        tests: list[dict[str, Any]],
        setup_code: str,
    ) -> tuple[float, float, float, float]:
        code = extract_python_code(str(raw), str(entry_point))
        cases = tuple(TestCase.from_dict(x) for x in tests)
        report = self.executor.run(
            code,
            entry_point=str(entry_point),
            cases=cases,
            setup_code=str(setup_code),
        )
        reward = compute_training_reward(
            syntax_ok=report.syntax_ok,
            public_pass_rate=report.pass_rate,
        )
        return (
            float(reward.total),
            float(reward.syntax),
            float(reward.public_pass_rate),
            float(reward.all_public_pass_bonus),
        )

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        completion_ids=None,
        task_id: list[str] | None = None,
        entry_point: list[str] | None = None,
        public_tests: list[list[dict[str, Any]]] | None = None,
        setup_code: list[str] | None = None,
        log_metric=None,
        log_extra=None,
        **kwargs,
    ) -> list[float]:
        if not (
            len(prompts)
            == len(completions)
            == len(task_id or [])
            == len(entry_point or [])
            == len(public_tests or [])
            == len(setup_code or [])
        ):
            raise ValueError("reward inputs must have one metadata row per completion")

        work_items = list(
            zip(
                completions,
                entry_point or [],
                public_tests or [],
                setup_code or [],
                strict=True,
            )
        )

        if self.max_workers == 1:
            scored = [
                self._score_one(raw, ep, tests, setup)
                for raw, ep, tests, setup in work_items
            ]
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                scored = list(
                    pool.map(
                        lambda item: self._score_one(*item),
                        work_items,
                    )
                )

        rewards = [x[0] for x in scored]
        syntax_values = [x[1] for x in scored]
        pass_rates = [x[2] for x in scored]
        all_pass_values = [x[3] for x in scored]

        if log_metric is not None and rewards:
            log_metric("reward/public_mean", sum(pass_rates) / len(pass_rates))
            log_metric("reward/syntax_mean", sum(syntax_values) / len(syntax_values))
            log_metric(
                "reward/public_all_pass_mean",
                sum(all_pass_values) / len(all_pass_values),
            )

        if log_extra is not None:
            log_extra("task_id", list(task_id or []))
            log_extra("public_pass_rate", pass_rates)

        return rewards


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("GRPO config must be a mapping")
    return data


def run_grpo(
    *,
    config_path: Path,
    tasks_path: Path,
    sft_adapter_path: Path,
    output_dir: Path | None = None,
    max_tasks: int | None = None,
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
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "GRPO dependencies are missing; install with "
            "pip install -e '.[model]'"
        ) from exc

    cfg = load_config(config_path)
    model_cfg = dict(cfg["model"])
    train_cfg = dict(cfg["training"])
    reward_cfg = dict(cfg["reward"])
    generation_cfg = dict(cfg["generation"])

    seed = int(train_cfg.get("seed", 42))
    rows = load_grpo_rows(
        tasks_path,
        seed=seed,
        max_tasks=max_tasks,
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
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        dtype=dtype,
        trust_remote_code=True,
    )
    base_model.config.use_cache = False

    if not sft_adapter_path.exists():
        raise FileNotFoundError(
            f"SFT adapter not found: {sft_adapter_path}. "
            "Run EXP-002 on Tang first."
        )

    model = PeftModel.from_pretrained(
        base_model,
        str(sft_adapter_path),
        is_trainable=True,
    )

    requested_max_steps = (
        int(max_steps)
        if max_steps is not None
        else int(train_cfg.get("max_steps", -1))
    )
    num_generations = int(generation_cfg["num_generations"])
    epochs = float(train_cfg.get("num_train_epochs", 1.0))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.0))
    per_device_train_batch_size = int(
        train_cfg["per_device_train_batch_size"]
    )
    gradient_accumulation_steps = int(
        train_cfg["gradient_accumulation_steps"]
    )
    warmup_steps = compute_warmup_steps(
        num_examples=len(dataset),
        num_generations=num_generations,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=requested_max_steps,
        num_train_epochs=epochs,
        warmup_ratio=warmup_ratio,
    )

    args = GRPOConfig(
        output_dir=str(final_output),
        num_train_epochs=epochs,
        max_steps=requested_max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=float(train_cfg["learning_rate"]),
        warmup_steps=warmup_steps,
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        lr_scheduler_type=str(
            train_cfg.get("lr_scheduler_type", "cosine")
        ),
        logging_steps=int(train_cfg.get("logging_steps", 1)),
        save_strategy=str(train_cfg.get("save_strategy", "no")),
        seed=seed,
        data_seed=seed,
        bf16=dtype_name == "bfloat16",
        fp16=dtype_name == "float16",
        gradient_checkpointing=bool(
            train_cfg.get("gradient_checkpointing", True)
        ),
        max_completion_length=int(
            generation_cfg["max_completion_length"]
        ),
        num_generations=num_generations,
        temperature=float(generation_cfg["temperature"]),
        top_p=float(generation_cfg["top_p"]),
        top_k=int(generation_cfg.get("top_k", 0)),
        beta=float(train_cfg.get("beta", 0.0)),
        loss_type=str(train_cfg.get("loss_type", "grpo")),
        scale_rewards=str(train_cfg.get("scale_rewards", "group")),
        mask_truncated_completions=bool(
            train_cfg.get("mask_truncated_completions", False)
        ),
        report_to="none",
        remove_unused_columns=False,
        use_vllm=False,
        log_completions=bool(train_cfg.get("log_completions", False)),
    )

    reward_func = VerifiableCodeReward(
        timeout_seconds=float(reward_cfg["timeout_seconds"]),
        memory_limit=str(reward_cfg["memory_limit"]),
        docker_image=str(reward_cfg["docker_image"]),
        max_workers=int(reward_cfg.get("max_workers", 1)),
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
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
        "tasks_path": str(tasks_path),
        "sft_adapter_path": str(sft_adapter_path),
        "output_dir": str(final_output),
        "model": model_name,
        "requested_model_revision": revision,
        "num_unique_tasks": len(dataset),
        "num_generations": num_generations,
        "num_train_epochs": epochs,
        "max_steps": requested_max_steps,
        "warmup_ratio_requested": warmup_ratio,
        "warmup_steps": warmup_steps,
        "seed": seed,
        "loss_type": args.loss_type,
        "beta": args.beta,
        "scale_rewards": args.scale_rewards,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
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
    parser = argparse.ArgumentParser(
        description="Run CodeRL-Lab verifiable-reward GRPO"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--max-steps", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_grpo(
        config_path=args.config,
        tasks_path=args.tasks,
        sft_adapter_path=args.sft_adapter,
        output_dir=args.output_dir,
        max_tasks=args.max_tasks,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
