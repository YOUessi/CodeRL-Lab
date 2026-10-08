from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from coderl_lab.train.posttrain_a_dpo_freeze import freeze_dpo_config, sha256_file


def fixture_bundle(tmp_path: Path) -> dict[str, Path]:
    cfg = {
        "experiment": {"id": "TRAIN-007C"},
        "model": {"name_or_path": "Qwen/Qwen3-1.7B-Base", "revision": "frozen_rev"},
        "initial_policy": {
            "sft_adapter": str(tmp_path / "train007a"),
            "expected_sha256": "REQUIRES_VERIFIED_TRAIN007A_CHECKPOINT",
        },
        "data": {
            "revision": "ultrafeedback_rev",
            "expected_train_pairs": 2,
            "expected_validation_pairs": 1,
        },
        "training": {"seed": 42},
    }
    template = tmp_path / "dpo.yaml"
    template.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    adapter_dir = tmp_path / "train007a"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"fake-model-weights-for-test")
    digest = sha256_file(adapter_dir / "adapter_model.safetensors")
    summary = tmp_path / "sft_summary.json"
    summary.write_text(json.dumps({
        "model": "Qwen/Qwen3-1.7B-Base",
        "requested_model_revision": "frozen_rev",
        "num_examples": 2,
        "quantization": {"mode": "nf4"},
        "saved_adapter_sha256": digest,
    }), encoding="utf-8")

    train_rows = [
        {"prompt": "prompt 1", "chosen": "good a", "rejected": "bad a"},
        {"prompt": "prompt 2", "chosen": "good b", "rejected": "bad b"},
    ]
    heldout_rows = [{"prompt": "prompt 3", "chosen": "good c", "rejected": "bad c"}]
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "validation.jsonl"
    for path, rows in ((train, train_rows), (valid, heldout_rows)):
        path.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "track": "A-posttraining", "stage": "dpo", "full_scan": True,
        "dataset_revision": "ultrafeedback_rev",
        "prompt_overlap": 0,
        "train": {"rows": 2, "sha256": sha256_file(train)},
        "validation": {"rows": 1, "sha256": sha256_file(valid)},
    }), encoding="utf-8")
    return {
        "template_path": template,
        "adapter_dir": adapter_dir,
        "sft_summary_path": summary,
        "manifest_path": manifest,
        "train_preferences_path": train,
        "validation_preferences_path": valid,
        "output_path": tmp_path / "runtime.yaml",
    }


def run_freeze(paths: dict[str, Path]):
    return freeze_dpo_config(**paths, expected_sft_train_examples=2)


def test_freeze_valid_qlora_adapter_and_disjoint_real_format(tmp_path: Path) -> None:
    paths = fixture_bundle(tmp_path)
    result = run_freeze(paths)
    assert result["validated_disjoint"] is True
    assert result["training_pairs"] == 2
    assert result["heldout_pairs"] == 1
    assert result["new_dpo_training_completed"] is False
    runtime = yaml.safe_load(paths["output_path"].read_text())
    assert runtime["initial_policy"]["expected_sha256"] == result["sft_adapter_sha256"]
    assert runtime["data"]["train_sha256"] == sha256_file(paths["train_preferences_path"])
    with pytest.raises(FileExistsError):
        run_freeze(paths)


def test_freeze_rejects_old_or_modified_adapter(tmp_path: Path) -> None:
    paths = fixture_bundle(tmp_path)
    (paths["adapter_dir"] / "adapter_model.safetensors").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="checkpoint SHA"):
        run_freeze(paths)
    assert not paths["output_path"].exists()


def test_freeze_rejects_smoke_manifest(tmp_path: Path) -> None:
    paths = fixture_bundle(tmp_path)
    m = json.loads(paths["manifest_path"].read_text())
    m["full_scan"] = False
    paths["manifest_path"].write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ValueError, match="smoke or contaminated"):
        run_freeze(paths)


def test_freeze_rejects_manifest_and_actual_data_mismatch(tmp_path: Path) -> None:
    paths = fixture_bundle(tmp_path)
    paths["train_preferences_path"].write_text('{"prompt": "fake"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="train SHA mismatch"):
        run_freeze(paths)


def test_freeze_rejects_train_validation_leak_even_if_hash_updated(tmp_path: Path) -> None:
    paths = fixture_bundle(tmp_path)
    train = json.loads(paths["train_preferences_path"].read_text().splitlines()[0])
    paths["validation_preferences_path"].write_text(json.dumps(train) + "\n", encoding="utf-8")
    m = json.loads(paths["manifest_path"].read_text())
    m["validation"]["sha256"] = sha256_file(paths["validation_preferences_path"])
    paths["manifest_path"].write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ValueError, match="prompt leakage"):
        run_freeze(paths)


def test_freeze_requires_real_4096_sample_sft_by_default(tmp_path: Path) -> None:
    paths = fixture_bundle(tmp_path)
    with pytest.raises(ValueError, match="4096-example"):
        freeze_dpo_config(**paths)
