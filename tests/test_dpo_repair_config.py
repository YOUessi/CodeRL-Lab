from pathlib import Path

import yaml

from coderl_lab.train.dpo import compute_warmup_steps, prepare_rows


def test_exp004b_dpo_config() -> None:
    cfg = yaml.safe_load(
        Path("configs/dpo_runtime_repair_qwen3_1.7b.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert cfg["model"]["name_or_path"] == "Qwen/Qwen3-1.7B-Base"
    assert cfg["training"]["beta"] == 0.1
    assert cfg["training"]["loss_type"] == "sigmoid"
    assert cfg["training"]["learning_rate"] == 5e-7
    assert cfg["training"]["per_device_train_batch_size"] == 1
    assert cfg["training"]["gradient_accumulation_steps"] == 8


def test_dpo_warmup_for_56_pairs() -> None:
    assert compute_warmup_steps(
        num_examples=56,
        epochs=3,
        per_device_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_ratio=0.05,
        max_steps=-1,
    ) == 2
