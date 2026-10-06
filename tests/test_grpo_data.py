import json
from pathlib import Path

from coderl_lab.train.grpo import compute_warmup_steps, load_grpo_rows


def test_grpo_loader_excludes_hidden_tests(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    row = {
        "task_id": "x",
        "prompt": "Implement add.",
        "entry_point": "add",
        "starter_code": "def add(a, b):\n    pass\n",
        "public_tests": [
            {
                "name": "public",
                "assertion": "assert add(1, 2) == 3",
            }
        ],
        "hidden_tests": [
            {
                "name": "hidden",
                "assertion": "assert add(-1, 1) == 0",
            }
        ],
        "setup_code": "",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    rows = load_grpo_rows(path, seed=42)
    assert len(rows) == 1
    assert "public_tests" in rows[0]
    assert "hidden_tests" not in rows[0]
    assert rows[0]["task_id"] == "x"


def test_grpo_warmup_for_smoke() -> None:
    assert compute_warmup_steps(
        num_examples=16,
        num_generations=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        max_steps=2,
        num_train_epochs=1,
        warmup_ratio=0.05,
    ) == 1


def test_grpo_warmup_for_full_mbpp_epoch() -> None:
    assert compute_warmup_steps(
        num_examples=374,
        num_generations=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        max_steps=-1,
        num_train_epochs=1,
        warmup_ratio=0.05,
    ) == 10


def test_grpo_config_pins_sft_adapter_hash() -> None:
    import yaml
    from pathlib import Path

    config = yaml.safe_load(
        Path("configs/grpo_qwen3_0.6b.yaml").read_text(encoding="utf-8")
    )
    assert (
        config["initial_policy"]["expected_sha256"]
        == "7fcb2b0ca7608be980fb4cbae5158241886ac67f3cc2f5b855cf34531c5982f7"
    )
