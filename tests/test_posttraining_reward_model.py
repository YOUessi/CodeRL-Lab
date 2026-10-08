from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml

from coderl_lab.train.reward_model import (
    compute_reward_metrics,
    file_sha256,
    pairwise_logistic_loss,
    read_pairs,
    render_pair,
    validate_preference_snapshot,
    validate_resume,
)


def preference(prompt: str, *, good: str = "An informed answer.", bad: str = "An incorrect answer."):
    return {
        "prompt": f"### User:\n{prompt}\n\n### Assistant:\n",
        "chosen": good+"\n",
        "rejected": bad+"\n",
    }


def bundle(root: Path):
    train = root / "train.jsonl"
    valid = root / "heldout.jsonl"
    train_rows = [preference("Explain GRPO"), preference("Explain DPO")]
    valid_rows = [preference("Explain supervised fine-tuning")]
    train.write_text("".join(json.dumps(x)+"\n" for x in train_rows))
    valid.write_text("".join(json.dumps(x)+"\n" for x in valid_rows))
    manifest = {
        "track":"A-posttraining", "stage":"dpo",
        "full_scan":True, "prompt_overlap":0, "dataset_revision":"H4_LOCK",
        "train":{"rows":2, "sha256":file_sha256(train)},
        "validation":{"rows":1, "sha256":file_sha256(valid)},
    }
    mpath = root / "manifest.json"
    mpath.write_text(json.dumps(manifest))
    cfg = {
        "experiment":{"id":"TRAIN-008A"},
        "model":{"revision":"QWEN_LOCK"},
        "data":{"revision":"H4_LOCK","expected_train_pairs":2,
                "expected_heldout_pairs":1},
        "training":{"seed":42},
    }
    cpath = root / "config.yaml"
    cpath.write_text(yaml.safe_dump(cfg))
    return {"config_path":cpath,"train_path":train,"heldout_path":valid,"manifest_path":mpath}


def test_pairwise_objective_is_numerically_stable_and_directional():
    assert pairwise_logistic_loss(1000) >= 0
    assert pairwise_logistic_loss(-1000) == pytest.approx(1000)
    assert pairwise_logistic_loss(2) < pairwise_logistic_loss(-2)
    metrics = compute_reward_metrics([2, -1, 0])
    assert metrics["pairs"] == 3
    assert metrics["preference_accuracy"] == pytest.approx(1/3)
    assert metrics["ties"] == 1
    assert metrics["mean_margin"] == pytest.approx(1/3)


def test_render_pair_changes_only_assistant_completion():
    row = preference("Define reward modeling")
    a,b = render_pair(row)
    assert a.startswith(row["prompt"])
    assert b.startswith(row["prompt"])
    assert a.endswith(row["chosen"]) and b.endswith(row["rejected"])
    assert a != b


def test_real_format_manifest_and_split_audit(tmp_path: Path):
    inputs = bundle(tmp_path)
    config, train, heldout, identity = validate_preference_snapshot(**inputs)
    assert len(train)==2 and len(heldout)==1
    assert identity["formal"] is True
    assert identity["train_file_sha256"] == file_sha256(inputs["train_path"])
    assert "QWEN_LOCK" == identity["model_revision"]


def test_preferences_cannot_include_identical_or_duplicate_responses(tmp_path: Path):
    path=tmp_path/"p.jsonl"
    row=preference("Duplicate prompt")
    path.write_text(json.dumps(row)+"\n"+json.dumps(row)+"\n")
    with pytest.raises(ValueError,match="duplicate"):
        read_pairs(path)
    row["rejected"]=row["chosen"]
    path.write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError,match="equal"):
        read_pairs(path)


def test_hidden_and_smoke_preference_leakage_rejected(tmp_path: Path):
    inputs=bundle(tmp_path)
    config, train, heldout, identity = validate_preference_snapshot(
        **inputs, train_limit=1, eval_limit=1,
    )
    assert len(train)==1
    assert identity["formal"] is False
    cfg=yaml.safe_load(inputs["config_path"].read_text())
    old=inputs["train_path"].read_text()
    with inputs["heldout_path"].open("w") as handle:
        handle.write(old.splitlines()[0]+"\n")
    manifest=json.loads(inputs["manifest_path"].read_text())
    manifest["validation"]["sha256"]=file_sha256(inputs["heldout_path"])
    inputs["manifest_path"].write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="prompt leakage"):
        validate_preference_snapshot(**inputs)


def test_manifest_sha_mutation_blocks_training(tmp_path: Path):
    inputs=bundle(tmp_path)
    inputs["train_path"].write_text(inputs["train_path"].read_text()+"\n")
    with pytest.raises(ValueError,match="manifest or hashes"):
        validate_preference_snapshot(**inputs)


def test_reward_checkpoint_resume_requires_frozen_identity_and_adapter(tmp_path: Path):
    root=tmp_path/"rm"
    root.mkdir()
    original={"experiment":"TRAIN-008A","train_file_sha256":"A","formal":True}
    (root/"training_identity.json").write_text(json.dumps(original))
    checkpoint=root/"checkpoint-32"
    checkpoint.mkdir()
    (checkpoint/"adapter_model.safetensors").write_bytes(b"adapter")
    (checkpoint/"state.pt").write_bytes(b"state")
    (checkpoint/"checkpoint_identity.json").write_text(json.dumps(original))
    assert validate_resume(
        output_dir=root,checkpoint=checkpoint,identity=original,
    )[0] == checkpoint.resolve()
    with pytest.raises(ValueError,match="identity mismatch"):
        validate_resume(output_dir=root,checkpoint=checkpoint,
                        identity={**original,"train_file_sha256":"ALTERED"})


def test_invalid_margin_metrics_fail_closed():
    with pytest.raises(ValueError,match="finite"):
        compute_reward_metrics([float("nan")])
    with pytest.raises(ValueError,match="finite"):
        compute_reward_metrics([])
