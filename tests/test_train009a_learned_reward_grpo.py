from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest
import yaml

from coderl_lab.train.learned_reward_grpo import (
    BoundedLearnedReward, assert_policy_prompt_token_budget, validate_grpo_contract, validate_frozen_models,
)
from coderl_lab.datasets import train009a_online_prompts as data


CFG=Path("configs/train009a_learned_reward_grpo.yaml")


def config():
    return yaml.safe_load(CFG.read_text(encoding="utf-8"))


def test_config_is_frozen_online_grpo_not_ppo_and_keeps_sft_kl():
    c=config()
    x=validate_grpo_contract(c,smoke=True)
    assert c["experiment"]["id"]=="TRAIN-009A"
    assert x["algorithm"].startswith("GRPO")
    assert x["kl_beta"]>0
    assert x["optimizer_steps"]==1
    assert x["num_generations"]==4
    assert c["reward_model"]["no_reward_model_parameter_updates"]
    assert c["data"]["exclude_reward_training_prompts"]
    assert c["data"]["exclude_reward_heldout_prompts"]
    assert c["training"]["reference_adapter"]=="ref"


def test_reject_kl_zero_reward_disclosure_and_prompt_incoherence():
    c=config()
    c["training"]["beta"]=0
    with pytest.raises(ValueError,match="invalid"):
        validate_grpo_contract(c,smoke=True)
    c=config()
    c["reward_model"]["max_prompt_tokens"]=200
    with pytest.raises(ValueError,match="invalid"):
        validate_grpo_contract(c,smoke=True)
    c=config()
    c["data"]["exclude_reward_heldout_prompts"]=False
    with pytest.raises(ValueError,match="invalid"):
        validate_grpo_contract(c,smoke=True)
    c=config()
    with pytest.raises(ValueError,match="frozen positive max_steps"):
        validate_grpo_contract(c,smoke=False)


def test_bounded_reward_uses_finite_frozen_scalar_model_and_never_raw_completions():
    seen=[]
    def fake_score(prompt,completion):
        seen.append((prompt,completion))
        return 2.0 if completion=="preferred" else -2.0
    scorer=BoundedLearnedReward(score_one=fake_score,temperature=2.0)
    out=scorer(prompts=["Sensitive question"]*4,
               completions=["preferred","rejected"," ","preferred"])
    assert out==pytest.approx([math.tanh(1),math.tanh(-1),-1,math.tanh(1)])
    assert len(seen)==3
    summary=scorer.aggregate()
    assert summary["rewarded_rollouts"]==4
    assert summary["empty_completions"]==1
    assert "Sensitive question" not in json.dumps(scorer.history)
    assert "preferred" not in json.dumps(scorer.history)
    assert summary["avg_bounded_reward_std"]>0


def test_reward_nonfinite_empty_inputs_fail_closed():
    scorer=BoundedLearnedReward(score_one=lambda a,b:float("nan"))
    with pytest.raises(FloatingPointError,match="NaN/Inf"):
        scorer(prompts=["P"],completions=["C"])
    with pytest.raises(ValueError,match="empty"):
        scorer(prompts=[],completions=[])
    with pytest.raises(ValueError,match="unequal"):
        scorer(prompts=["P"],completions=["C","D"])
    with pytest.raises(ValueError,match="online rollout"):
        scorer(prompts=[""],completions=["C"])


def fixture_dataset(tmp_path:Path, monkeypatch):
    root=tmp_path/"H4"
    root.mkdir()
    sft=[{"prompt":f"### User:\nQuestion {i}\n\n### Assistant:\n",
          "response":"supervised answer", "metadata":{"source_split":"train_sft"}} for i in range(4096)]
    sft_val=[{"prompt":f"### User:\nValidation {i}\n\n### Assistant:\n","response":"validation answer"}
             for i in range(256)]
    dpo=[{"prompt":f"### User:\nRM training {i}\n\n### Assistant:\n",
          "chosen":"X","rejected":"Y"} for i in range(2048)]
    dpo_val=[{"prompt":f"### User:\nRM heldout {i}\n\n### Assistant:\n",
              "chosen":"X","rejected":"Y"} for i in range(256)]
    # Deliberate overlaps with learned RM source train AND heldout; exclude.
    sft[30]["prompt"]=dpo[111]["prompt"]
    sft[55]["prompt"]=dpo_val[131]["prompt"]
    files={"sft_train.jsonl":sft,"sft_validation.jsonl":sft_val,
           "dpo_train.jsonl":dpo,"dpo_validation.jsonl":dpo_val}
    fingerprints={}
    for filename,rows in files.items():
        path=root/filename
        path.write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in rows),encoding="utf-8")
        fingerprints[filename]=data.digest_file(path)
    for key,f in [
        ("SFT_TRAIN_SHA","sft_train.jsonl"),("SFT_VALID_SHA","sft_validation.jsonl"),
        ("RM_TRAIN_SHA","dpo_train.jsonl"),("RM_VALID_SHA","dpo_validation.jsonl"),
    ]:
        monkeypatch.setattr(data,key,fingerprints[f])
    m1={"track":"A-posttraining","stage":"sft","full_scan":True,
        "dataset_revision":data.ULTRACHAT_REV,"prompt_overlap":0,
        "train":{"rows":4096,"sha256":fingerprints["sft_train.jsonl"]},
        "validation":{"rows":256,"sha256":fingerprints["sft_validation.jsonl"]}}
    m2={"track":"A-posttraining","stage":"dpo","full_scan":True,
        "dataset_revision":data.ULTRAFEEDBACK_REV,"prompt_overlap":0,
        "train":{"rows":2048,"sha256":fingerprints["dpo_train.jsonl"]},
        "validation":{"rows":256,"sha256":fingerprints["dpo_validation.jsonl"]}}
    (root/"sft_manifest.json").write_text(json.dumps(m1))
    (root/"dpo_manifest.json").write_text(json.dumps(m2))
    return root,dpo[111]["prompt"],dpo_val[131]["prompt"]


def test_cross_domain_disjoint_online_prompt_pool(tmp_path:Path,monkeypatch):
    root,bad_rm_train,bad_rm_valid=fixture_dataset(tmp_path,monkeypatch)
    selected,manifest=data.frozen_prompt_pool(root=root,max_train_prompts=16)
    assert len(selected)==16
    assert manifest["eligible_train_prompts"]==4094
    assert manifest["discarded_train_due_to_rm_any_split_overlap"]==2
    assert bad_rm_train not in [x["prompt"] for x in selected]
    assert bad_rm_valid not in [x["prompt"] for x in selected]
    assert not any("response" in x for x in selected)
    again,audit=data.frozen_prompt_pool(root=root,max_train_prompts=16)
    assert again==selected and manifest==audit
    with pytest.raises(ValueError,match="invalid|max_train_prompts"):
        data.frozen_prompt_pool(root=root,max_train_prompts=0)


def test_rlhf_input_hash_mutation_detected_before_sampling(tmp_path:Path,monkeypatch):
    root,_,_=fixture_dataset(tmp_path,monkeypatch)
    path=root/"dpo_validation.jsonl"
    path.write_text(path.read_text()+"\n")
    with pytest.raises(ValueError,match="SHA mismatch"):
        data.frozen_prompt_pool(root=root,max_train_prompts=16)


def test_rm_reward_identity_from_formal_manifest(tmp_path:Path,monkeypatch):
    from coderl_lab.train import learned_reward_grpo as online
    cfg=config()
    policy=tmp_path/"SFT";reward=tmp_path/"Reward"
    policy.mkdir();reward.mkdir()
    (policy/"adapter_model.safetensors").write_bytes(b"sft")
    (reward/"adapter_model.safetensors").write_bytes(b"trained-reward")
    monkeypatch.setattr(online,"SFT_SHA",online.sha256(policy/"adapter_model.safetensors"))
    monkeypatch.setattr(online,"RM_SHA",online.sha256(reward/"adapter_model.safetensors"))
    cfg["initial_policy"]["expected_sha256"]=online.SFT_SHA
    cfg["reward_model"]["expected_sha256"]=online.RM_SHA
    sft={"model":online.MODEL,"requested_model_revision":online.REVISION,
         "num_examples":4096,"quantization":{"mode":"nf4"},
         "saved_adapter_sha256":online.SFT_SHA}
    rm={"experiment":"TRAIN-008A","status":"formal","optimizer_steps":256,
        "train_pairs":2048,"heldout_pairs":256,
        "training_identity":{"formal":True,"train_file_sha256":cfg["reward_model"]["source_train_sha256"],
        "heldout_file_sha256":cfg["reward_model"]["source_heldout_sha256"]},
        "adapter_sha256":online.RM_SHA,
        "heldout":{"preference_accuracy":0.56640625}}
    (policy/"run_summary.json").write_text(json.dumps(sft))
    (reward/"run_summary.json").write_text(json.dumps(rm))
    identity=validate_frozen_models(config=cfg,policy_dir=policy,reward_dir=reward)
    assert identity["reward_validation_accuracy"]==pytest.approx(0.56640625)
    (reward/"adapter_model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="provenance"):
        validate_frozen_models(config=cfg,policy_dir=policy,reward_dir=reward)


class _FakeTokenizer:
    def __call__(self,*,text:str):
        return {"input_ids": list(range(len(text)))}


def test_actual_policy_prompt_budget_blocks_removed_grpo_config_keyword():
    rows=[{"prompt":"P"*13},{"prompt":"A"*8},{"prompt":"B"*16}]
    audit=assert_policy_prompt_token_budget(
        rows,_FakeTokenizer(),max_prompt_tokens=20,
    )
    assert audit["max_length_tokens"]==16
    assert audit["silently_truncated"] is False
    assert audit["overlong_prompts"]==0
    with pytest.raises(ValueError,match="exceed explicit"):
        assert_policy_prompt_token_budget(
            rows,_FakeTokenizer(),max_prompt_tokens=15,
        )


def test_smoke_v2_does_not_override_recorded_failed_v1_identity():
    shell=Path("scripts/run_train009a_learned_reward_grpo.sh").read_text()
    assert "train009a-reward-grpo-smoke-v2" in shell
    assert 'test ! -e "$OUTPUT/training_identity.json"' in shell
    code=Path("src/coderl_lab/train/learned_reward_grpo.py").read_text()
    assert "max_prompt_length=int(g[" not in code
    assert "assert_policy_prompt_token_budget(" in code


def test_separate_v3_smoke_allows_real_gradients_but_formal_still_masks():
    original=config()
    v3=yaml.safe_load(Path("configs/train009a_learned_reward_grpo_smoke_v3.yaml").read_text())
    assert original["training"]["mask_truncated_completions"] is True
    assert v3["training"]["mask_truncated_completions"] is False
    assert v3["training"]["gradient_smoke_allow_truncated"] is True
    assert v3["generation"]["max_completion_length"]==256
    assert v3["generation"]["max_prompt_length"]==480
    assert validate_grpo_contract(v3,smoke=True)["optimizer_steps"]==2
    assert validate_grpo_contract(v3,smoke=True)["kl_beta"]==0.02
    assert v3["initial_policy"]==original["initial_policy"]
    assert v3["reward_model"]==original["reward_model"]
    with pytest.raises(ValueError,match="invalid"):
        validate_grpo_contract(v3,smoke=False)
    script=Path("scripts/run_train009a_learned_reward_grpo_smoke_v3.sh").read_text()
    assert "train009a-reward-grpo-smoke-v3" in script
    assert 's["policy_weights_actually_changed"] is True' in script
    assert 's["at_least_one_real_nonzero_policy_gradient"] is True' in script
    assert 's["frozen_reward_adapter_unchanged"] is True' in script
