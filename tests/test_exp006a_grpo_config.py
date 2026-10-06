from pathlib import Path

import yaml


def test_exp006a_grpo_is_pure_and_pins_sft_adapter() -> None:
    config = yaml.safe_load(
        Path("configs/grpo_qwen3_1.7b.yaml").read_text(encoding="utf-8")
    )
    assert config["model"]["name_or_path"] == "Qwen/Qwen3-1.7B-Base"
    assert (
        config["model"]["revision"]
        == "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
    )
    assert config["initial_policy"]["expected_sha256"] == (
        "e1ea9a007a3e4213cdeb1ca6b1e12d29adb83594991421757d0d40b9168d0861"
    )
    assert config["generation"]["num_generations"] == 4
    assert config["training"]["loss_type"] == "grpo"
    assert config["training"]["beta"] == 0.0
    assert config["training"]["per_device_train_batch_size"] == 4
    assert config["training"]["gradient_accumulation_steps"] == 2
    assert config["reward"]["hidden_tests_allowed"] is False
