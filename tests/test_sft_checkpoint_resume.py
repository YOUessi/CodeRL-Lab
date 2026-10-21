from __future__ import annotations

import json
from pathlib import Path

import pytest

from coderl_lab.train.sft import (
    build_training_identity,
    initialize_training_identity,
    validate_resume_checkpoint,
)


def _inputs(root: Path):
    cfg = root / "config.yaml"
    train = root / "train.jsonl"
    validation = root / "validation.jsonl"
    cfg.write_text("save_strategy: steps\nsave_steps: 32\n")
    train.write_text('{"sample":"train"}\n')
    validation.write_text('{"sample":"validation"}\n')
    return cfg, train, validation


def _identity(root: Path):
    cfg, train, validation = _inputs(root)
    return build_training_identity(
        config_path=cfg, data_path=train,
        heldout_path=validation, train_rows=4096, heldout_rows=256,
    )


def _checkpoint(output: Path, step: int) -> Path:
    ckpt = output / f"checkpoint-{step}"
    ckpt.mkdir()
    for file in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (ckpt / file).write_bytes(b"test state")
    (ckpt / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    return ckpt


def test_new_training_writes_frozen_identity_and_accepts_matching_checkpoint(tmp_path: Path):
    frozen = _identity(tmp_path)
    output = tmp_path / "run"
    assert initialize_training_identity(
        output_dir=output, identity=frozen, resume_from_checkpoint=None,
    ) is None
    ckpt = _checkpoint(output, 32)
    assert validate_resume_checkpoint(
        checkpoint=ckpt, output_dir=output, identity=frozen,
    ) == ckpt.resolve()


def test_existing_run_cannot_restart_without_explicit_resume(tmp_path: Path):
    frozen = _identity(tmp_path)
    output = tmp_path / "run"
    initialize_training_identity(
        output_dir=output, identity=frozen, resume_from_checkpoint=None,
    )
    with pytest.raises(FileExistsError, match="identity already exists"):
        initialize_training_identity(
            output_dir=output, identity=frozen, resume_from_checkpoint=None,
        )


def test_changed_training_data_or_config_blocks_resume(tmp_path: Path):
    cfg, train, validation = _inputs(tmp_path)
    identity = build_training_identity(
        config_path=cfg, data_path=train, heldout_path=validation,
        train_rows=4096, heldout_rows=256,
    )
    output = tmp_path / "run"
    initialize_training_identity(
        output_dir=output, identity=identity, resume_from_checkpoint=None,
    )
    ckpt = _checkpoint(output, 32)
    train.write_text('{"sample":"changed train data"}\n')
    different = build_training_identity(
        config_path=cfg, data_path=train, heldout_path=validation,
        train_rows=4096, heldout_rows=256,
    )
    with pytest.raises(ValueError, match="SHA mismatch"):
        validate_resume_checkpoint(
            checkpoint=ckpt, output_dir=output, identity=different,
        )


def test_optimizer_state_required_and_cross_run_path_forbidden(tmp_path: Path):
    identity = _identity(tmp_path)
    output = tmp_path / "run"
    initialize_training_identity(
        output_dir=output, identity=identity, resume_from_checkpoint=None,
    )
    ckpt = _checkpoint(output, 32)
    (ckpt / "optimizer.pt").unlink()
    with pytest.raises(FileNotFoundError, match="optimizer.pt"):
        validate_resume_checkpoint(
            checkpoint=ckpt, output_dir=output, identity=identity,
        )
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="belong"):
        validate_resume_checkpoint(
            checkpoint=ckpt, output_dir=other, identity=identity,
        )


def test_checkpoint_step_state_mismatch_rejected(tmp_path: Path):
    identity = _identity(tmp_path)
    output = tmp_path / "run"
    initialize_training_identity(
        output_dir=output, identity=identity, resume_from_checkpoint=None,
    )
    ckpt = _checkpoint(output, 32)
    (ckpt / "trainer_state.json").write_text(json.dumps({"global_step": 31}))
    with pytest.raises(ValueError, match="step checkpoint"):
        validate_resume_checkpoint(
            checkpoint=ckpt, output_dir=output, identity=identity,
        )
