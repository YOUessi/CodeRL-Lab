"""CPU contract tests for the policy-likelihood matched preference baseline."""
from __future__ import annotations

import copy
import math

import pytest

from coderl_lab.analysis.train007d_preference_eval import (
    MAX_LENGTH, adapter_weights_path, encode_pair, paired_bootstrap, sha, summarize,
)


class DummyTokenizer:
    eos_token="<eos>"
    def __call__(self,*,text:str):
        return {"input_ids":[ord(c) for c in text]}


def test_pair_packing_preserves_same_prompt_and_never_scores_prompt_as_completion():
    tokenizer=DummyTokenizer()
    row={"prompt":"Q"*100,"chosen":" good","rejected":" bad"}
    v=encode_pair(tokenizer,row,max_length=128)
    assert v is not None
    assert v["prompt_token_count"]==100
    for arm in ("chosen","rejected"):
        assert v[arm+"_mask"][:100]==[False]*100
        assert all(v[arm+"_mask"][100:])
        assert len(v[arm+"_mask"])==len(v[arm+"_ids"])==v["prompt_token_count"]+v[arm+"_length"]
        assert v[arm+"_length"]>0
    assert v["chosen_ids"][:100]==v["rejected_ids"][:100]


def test_filtered_prompts_match_trl_keep_start_rule():
    tokenizer=DummyTokenizer()
    x={"prompt":"P"*MAX_LENGTH,"chosen":"A","rejected":"B"}
    assert encode_pair(tokenizer,x) is None
    x["prompt"]="P"*(MAX_LENGTH-1)
    v=encode_pair(tokenizer,x)
    assert v is not None
    assert v["chosen_length"]==1
    assert v["rejected_length"]==1
    assert v["chosen_was_truncated"] is True


def test_paired_accuracy_metrics_and_no_standalone_weak_smoke_heldout():
    rows=[]
    for i in range(252):
        rows.append({
            "row_id":i,
            "sft_mean_delta":-1.0 if i<200 else 1.0,
            "dpo_mean_delta":1.0 if i<201 else -1.0,
            "sft_total_delta":-2.0 if i<200 else 2.0,
            "dpo_total_delta":2.0 if i<201 else -2.0,
        })
    result=summarize(rows)
    assert result["raw_validation_pairs"]==256
    assert result["effective_validation_pairs"]==252
    assert result["sft"]["correct"]==52
    assert result["dpo"]["correct"]==201
    assert result["paired_accuracy_flips"]["sft_wrong_to_dpo_correct"]==200
    assert result["paired_accuracy_flips"]["sft_correct_to_dpo_wrong"]==51
    assert result["paired_mean_token_preference_accuracy_difference"]["observed_difference"]==pytest.approx(149/252)


def test_summary_rejects_partial_results_duplicates_and_invalid_numbers():
    rows=[{"row_id":i,"sft_mean_delta":1.0,
           "dpo_mean_delta":1.0,"sft_total_delta":1.0,
           "dpo_total_delta":1.0} for i in range(252)]
    with pytest.raises(ValueError,match="all 252"):
        summarize(rows[:251])
    duplicate=copy.deepcopy(rows)
    duplicate[-1]["row_id"]=0
    with pytest.raises(ValueError,match="duplicate"):
        summarize(duplicate)
    invalid=copy.deepcopy(rows)
    invalid[44]["dpo_mean_delta"]=float("nan")
    with pytest.raises(ValueError,match="nonfinite"):
        summarize(invalid)


def test_bootstrap_is_paired_reproducible_and_not_per_arm_unpaired():
    deltas=[1,0,-1,1,0,0,0,1,0,-1]
    x=paired_bootstrap(deltas,seed=42,iterations=2000)
    y=paired_bootstrap(deltas,seed=42,iterations=2000)
    assert x==y
    assert x["observed_difference"]==pytest.approx(0.1)
    assert x["ci95_low"]<=x["observed_difference"]<=x["ci95_high"]
    assert paired_bootstrap([0]*8,seed=42,iterations=500)["ci95_high"]==0
    with pytest.raises(ValueError):
        paired_bootstrap([4,-1])


def test_peft_adapter_hash_resolves_directory_and_file_equivalently(tmp_path):
    directory=tmp_path/"trained_sft"
    directory.mkdir()
    weights=directory/"adapter_model.safetensors"
    weights.write_bytes(b"valid adapter fixture")
    assert adapter_weights_path(directory)==weights
    assert adapter_weights_path(weights)==weights
    assert sha(adapter_weights_path(directory))==sha(adapter_weights_path(weights))
    with pytest.raises(FileNotFoundError,match="adapter_model.safetensors"):
        adapter_weights_path(tmp_path/"missing")


def test_peft_adapter_hash_rejects_arbitrary_directory_and_wrong_file(tmp_path):
    directory=tmp_path/"empty"
    directory.mkdir()
    with pytest.raises(FileNotFoundError,match="adapter_model.safetensors"):
        adapter_weights_path(directory)
    arbitrary=tmp_path/"model.bin"
    arbitrary.write_bytes(b"not the PEFT frozen adapter")
    with pytest.raises(FileNotFoundError,match="adapter_model.safetensors"):
        adapter_weights_path(arbitrary)
