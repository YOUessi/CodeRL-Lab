from pathlib import Path

import yaml


def test_exp006a_model_revision_and_effective_batch() -> None:
    config = yaml.safe_load(
        Path("configs/sft_qwen3_1.7b.yaml").read_text(encoding="utf-8")
    )
    assert config["model"]["name_or_path"] == "Qwen/Qwen3-1.7B-Base"
    assert (
        config["model"]["revision"]
        == "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
    )
    train = config["training"]
    assert (
        train["per_device_train_batch_size"]
        * train["gradient_accumulation_steps"]
        == 16
    )
