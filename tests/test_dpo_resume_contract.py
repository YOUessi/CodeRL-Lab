from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from coderl_lab.train.dpo import parse_args, initialize_dpo_training_identity
from coderl_lab.train.sft import build_training_identity, initialize_training_identity


def test_train007c_dpo_uses_step32_rotating_checkpoints() -> None:
    config = yaml.safe_load(
        Path("configs/posttrain_a_dpo_qwen3_1.7b.yaml").read_text(encoding="utf-8")
    )
    assert config["experiment"]["id"] == "TRAIN-007C"
    assert config["quantization"]["mode"] == "nf4"
    assert config["training"]["save_strategy"] == "steps"
    assert config["training"]["save_steps"] == 32
    assert config["training"]["save_total_limit"] >= 2
    assert config["initial_policy"]["expected_sha256"] == "REQUIRES_VERIFIED_TRAIN007A_CHECKPOINT"


def test_dpo_cli_has_explicit_resumption_option(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", [
        "dpo", "--config", "config.yaml",
        "--preferences", "train.jsonl",
        "--sft-adapter", "trained_adapter",
        "--resume-from-checkpoint", "run/checkpoint-64",
    ])
    args = parse_args()
    assert args.resume_from_checkpoint == Path("run/checkpoint-64")


def test_frozen_dpo_identity_includes_adapter_weights_and_preference_data(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    train = tmp_path / "prefs.jsonl"
    validation = tmp_path / "validation.jsonl"
    config.write_text("id: TRAIN-007C\n", encoding="utf-8")
    train.write_text('{"prompt":"A","chosen":"yes","rejected":"no"}\n')
    validation.write_text('{"prompt":"B","chosen":"yes","rejected":"no"}\n')
    identity = build_training_identity(
        config_path=config,
        data_path=train,
        heldout_path=validation,
        train_rows=2048,
        heldout_rows=256,
    )
    identity["source_sft_adapter_sha256"] = "SFT_FROZEN_SHA"
    output = tmp_path / "train007c"
    initialize_training_identity(
        output_dir=output,
        identity=identity,
        resume_from_checkpoint=None,
    )
    p = output / "checkpoint-32"
    p.mkdir()
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (p / name).write_bytes(b"checkpoint-test")
    (p / "trainer_state.json").write_text(json.dumps({"global_step": 32}))
    changed = dict(identity)
    changed["source_sft_adapter_sha256"] = "DIFFERENT_SFT_ADAPTER"
    with pytest.raises(ValueError, match="SHA mismatch"):
        initialize_training_identity(
            output_dir=output,
            identity=changed,
            resume_from_checkpoint=p,
        )
    assert initialize_training_identity(
        output_dir=output, identity=identity,
        resume_from_checkpoint=p,
    ) == p.resolve()


def test_safe_retry_exact_two_files_preserves_bytes(tmp_path: Path) -> None:
    root = tmp_path / "dpo"
    root.mkdir()
    cfg = root / "frozen_config.yaml"
    cfg.write_text("id: TRAIN-007C\n", encoding="utf-8")
    from hashlib import sha256
    identity = {"config_sha256": sha256(cfg.read_bytes()).hexdigest(),
                "source_sft_adapter_sha256": "PINNED",
                "train_rows": 32, "heldout_rows": 16}
    recorded = root / "training_identity.json"
    recorded.write_text(json.dumps(identity))
    before = [cfg.read_bytes(), recorded.read_bytes()]
    assert initialize_dpo_training_identity(
        output_dir=root, identity=identity, resume_from_checkpoint=None,
        retry_pretraining_failure=True,
    ) is None
    assert [cfg.read_bytes(), recorded.read_bytes()] == before


def test_safe_retry_rejects_accidental_optimizer_or_changed_identity(tmp_path: Path) -> None:
    from hashlib import sha256
    root=tmp_path / "dpo"
    root.mkdir()
    cfg=root/"frozen_config.yaml"
    cfg.write_text("id: TRAIN-007C\n")
    identity={"config_sha256":sha256(cfg.read_bytes()).hexdigest(),
              "source_sft_adapter_sha256":"PINNED"}
    (root/"training_identity.json").write_text(json.dumps(identity))
    with pytest.raises(ValueError,match="identity mismatch"):
        initialize_dpo_training_identity(
          output_dir=root,identity={**identity,"source_sft_adapter_sha256":"BAD"},
          resume_from_checkpoint=None,retry_pretraining_failure=True)
    (root/"optimizer.pt").write_bytes(b"some optimizer data")
    with pytest.raises(ValueError,match="unexpected files"):
        initialize_dpo_training_identity(
          output_dir=root,identity=identity,
          resume_from_checkpoint=None,retry_pretraining_failure=True)


def test_pretraining_retry_and_optimizer_resume_mutually_exclusive(tmp_path: Path):
    with pytest.raises(ValueError, match="exclusive"):
        initialize_dpo_training_identity(
            output_dir=tmp_path,identity={},
            resume_from_checkpoint=tmp_path/"checkpoint-32",
            retry_pretraining_failure=True,
        )
