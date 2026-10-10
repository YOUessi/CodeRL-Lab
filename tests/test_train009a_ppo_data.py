from __future__ import annotations

import hashlib
import copy
from pathlib import Path

import pytest

from coderl_lab.train.ppo_data import choose_train_prompts, sha_prompt
from coderl_lab.train.ppo_online import (
    frozen_provenance, load_config, SFT_SHA, REWARD_SHA,
)


def row(text: str):
    prompt=f"### User:\n{text}\n\n### Assistant:\n"
    return {
        "format":"raw","prompt":prompt,
        "metadata":{"prompt_sha256":sha_prompt(prompt)},
    }


def test_train_only_selection_is_stable_under_source_row_order():
    source=[row(f"user request {i}") for i in range(14)]
    excluded={sha_prompt(source[3]["prompt"]),sha_prompt(source[8]["prompt"])}
    a=choose_train_prompts(rows=source,excluded_prompt_sha256=excluded,n=8,seed=42)
    b=choose_train_prompts(rows=list(reversed(source)),excluded_prompt_sha256=excluded,n=8,seed=42)
    assert a==b
    assert all(x["prompt_sha256"] not in excluded for x in a)
    assert len({x["prompt_sha256"] for x in a})==8
    assert all(x["prompt"].rstrip().endswith("### Assistant:") for x in a)


def test_dedup_and_malformed_prompts_fail_closed():
    a=row("valid task")
    with pytest.raises(ValueError,match="duplicate"):
        choose_train_prompts(rows=[a,a],excluded_prompt_sha256=set(),n=1,seed=42)
    changed=copy.deepcopy(a)
    changed["metadata"]["prompt_sha256"]="bad"
    with pytest.raises(ValueError,match="SHA"):
        choose_train_prompts(rows=[changed],excluded_prompt_sha256=set(),n=1,seed=42)
    with pytest.raises(ValueError,match="insufficient"):
        choose_train_prompts(rows=[a],excluded_prompt_sha256={sha_prompt(a["prompt"])},n=1,seed=42)


def test_ppo_preregistered_training_budget_and_separate_reference_policy():
    cfg=load_config(Path("configs/train009a_ppo_clip_rlhf_smoke.yaml"))
    assert cfg["experiment"]["id"]=="TRAIN-009A"
    assert cfg["policy"]["expected_adapter_sha256"]==SFT_SHA
    assert cfg["reward"]["expected_adapter_sha256"]==REWARD_SHA
    assert cfg["policy"]["reference"]=="frozen_identical_sft_adapter"
    assert cfg["policy"]["update"]=="policy_lora_plus_scalar_value_head"
    assert cfg["rollout"]["prompts_per_step"]==2
    assert cfg["rollout"]["rollouts_per_prompt"]==2
    assert cfg["ppo"]["max_steps"]==4
    assert cfg["ppo"]["epochs_per_batch"]==2
    assert cfg["output"]["no_training_on_validation_labels"] is True
    assert cfg["data"]["n_rollout_prompts"]==8
    assert cfg["data"]["n_probe_prompts"]==4


def test_source_verification_refuses_missing_adapters(tmp_path: Path):
    cfg=Path("configs/train009a_ppo_clip_rlhf_smoke.yaml")
    with pytest.raises(FileNotFoundError):
        frozen_provenance(
            config_path=cfg,
            data_root=tmp_path,
            sft_dir=tmp_path/"missing-sft",
            reward_dir=tmp_path/"missing-rm",
        )


def test_script_uses_shared_gpu_mutex_no_hidden_tests():
    script=Path("scripts/run_train009a_ppo_clip_smoke.sh").read_text()
    assert "flock -n 9" in script
    assert "gpu_preflight --min-free-mib 11000" in script
    assert "QLORA_EXTRA_PYTHONPATH" in script
    assert "no_heldout_preference_labels_used_for_policy_gradient" in script
    assert "expected RM" not in script or "reward" in script.lower()
