from pathlib import Path

import yaml


def test_randomized_label_control_matches_semantic_dpo_budget() -> None:
    cfg = yaml.safe_load(
        Path(
            "configs/dpo_runtime_repair_random_control_qwen3_1.7b.yaml"
        ).read_text(encoding="utf-8")
    )
    assert cfg["model"]["name_or_path"] == "Qwen/Qwen3-1.7B-Base"
    assert cfg["data"]["label_seed"] == 42042
    assert cfg["data"]["swap_fraction"] == 0.5
    assert cfg["training"]["seed"] == 42
    assert cfg["training"]["num_train_epochs"] == 3
    assert cfg["training"]["gradient_accumulation_steps"] == 8
    assert cfg["training"]["learning_rate"] == 5e-7
    assert cfg["training"]["beta"] == 0.1
    assert cfg["training"]["loss_type"] == "sigmoid"
