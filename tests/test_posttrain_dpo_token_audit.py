from __future__ import annotations

import json
from pathlib import Path
import pytest
import yaml

from coderl_lab.analysis.posttrain_dpo_token_audit import (
    audit_split, audit, sha256,
)


class LengthTokenizer:
    def __call__(self, *, text: str):
        return {"input_ids": list(range(len(text)))}


def _pair(n):
    return {"prompt":"x"*n,"chosen":"correct","rejected":"incorrect"}


def test_trl_keep_start_drops_prompt_that_fills_entire_limit():
    x=audit_split(
        [_pair(8),_pair(10),_pair(13)],tokenizer=LengthTokenizer(),
        max_length=10,
    )
    assert x["raw_pairs"]==3
    assert x["kept_pairs_after_keep_start_prompt_filter"]==1
    assert x["dropped_pairs_due_to_prompt_filling_max_length"]==2
    assert x["dropped_sample_indices"]==[1,2]
    assert x["kept_fraction"]==pytest.approx(1/3)


def test_no_label_or_hidden_correctness_required():
    x=audit_split(
        [_pair(9),_pair(9)],tokenizer=LengthTokenizer(),max_length=10,
    )
    assert x["dropped_pairs_due_to_prompt_filling_max_length"]==0


def test_frozen_manifest_guards_source_and_predicts_steps(tmp_path: Path):
    train=tmp_path/"train.jsonl"
    valid=tmp_path/"valid.jsonl"
    train.write_text("\n".join(json.dumps(_pair(n)) for n in (8,10,13,9))+"\n")
    valid.write_text(json.dumps(_pair(9))+"\n")
    config=tmp_path/"cfg.yaml"
    cfg={
        "experiment":{"id":"TRAIN-007C"},
        "model":{"name_or_path":"Qwen/Qwen3-1.7B-Base",
                 "revision":"ea980cb0a6c2ae4b936e82123acc929f1cec04c1"},
        "data":{"revision":"SOURCE","expected_train_pairs":4,
                "expected_validation_pairs":1},
        "training":{"max_length":10,"per_device_train_batch_size":1,
                    "gradient_accumulation_steps":2},
    }
    config.write_text(yaml.safe_dump(cfg))
    manifest=tmp_path/"manifest.json"
    m={"track":"A-posttraining","stage":"dpo","full_scan":True,
       "dataset_revision":"SOURCE","prompt_overlap":0,
       "train":{"rows":4,"sha256":sha256(train)},
       "validation":{"rows":1,"sha256":sha256(valid)}}
    manifest.write_text(json.dumps(m))
    r=audit(config=config,manifest=manifest,train=train,heldout=valid,
            tokenizer=LengthTokenizer())
    assert r["train"]["kept_pairs_after_keep_start_prompt_filter"]==2
    assert r["expected_optimizer_steps_one_epoch"]==1
    assert r["new_cuda_training"] is False
    train.write_text(train.read_text()+"\n")
    with pytest.raises(ValueError,match="mismatch"):
        audit(config=config,manifest=manifest,train=train,heldout=valid,
              tokenizer=LengthTokenizer())
