from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from coderl_lab.train.train012_preflight import (
    MODEL, REVISION, SFT_SHA, DPO_TRAIN_SHA, DPO_HELDOUT_SHA,
    validate, validate_records,
)


def fixture():
    cfg=yaml.safe_load(
        Path("configs/train012_7b_qlora_dpo_ultrafeedback.yaml").read_text()
    )
    manifest={
        "stage":"dpo", "full_scan":True, "prompt_overlap":0,
        "dataset_revision":cfg["data"]["revision"],
        "train":{"rows":2048,"sha256":DPO_TRAIN_SHA},
        "validation":{"rows":256,"sha256":DPO_HELDOUT_SHA},
    }
    sft={
        "saved_adapter_sha256":SFT_SHA,"model":MODEL,
        "requested_model_revision":REVISION,
        "num_examples":4096,"num_train_epochs":1.0,
        "quantization":{
            "mode":"nf4","double_quant":True,"compute_dtype":"bfloat16",
        },
    }
    kwargs={
        "cfg":cfg, "train_manifest":manifest, "sft_summary":sft,
        "sft_sha":SFT_SHA,"train_sha":DPO_TRAIN_SHA,
        "eval_sha":DPO_HELDOUT_SHA,
        "train_prompts":{f"train_{i}" for i in range(2048)},
        "eval_prompts":{f"eval_{i}" for i in range(256)},
    }
    return kwargs


def test_frozen_7b_sft_and_disjoint_ultrafeedback_accepted():
    accepted=validate(**fixture())
    assert accepted["status"]=="cpu_preflight_passed"
    assert accepted["prompts_disjoint"] is True
    assert accepted["train_pairs"]==2048
    assert accepted["heldout_pairs"]==256
    assert accepted["new_7b_dpo_gpu_training_completed"] is False


@pytest.mark.parametrize("change",["adapter","data","overlap","model","beta","quant"])
def test_freeze_rejects_invalid_inputs(change):
    k=fixture()
    if change=="adapter": k["sft_sha"]="tampered"
    if change=="data": k["train_sha"]="tampered"
    if change=="overlap": k["eval_prompts"].add("train_20")
    if change=="model": k["cfg"]["model"]["name_or_path"]="other-model"
    if change=="beta": k["cfg"]["training"]["beta"]=1.0
    if change=="quant": k["cfg"]["quantization"]["mode"]="none"
    with pytest.raises(ValueError):
        validate(**k)


def test_duplicate_preference_prompt_rejected(tmp_path):
    file=tmp_path/"data.jsonl"
    file.write_text(
        '{"prompt":"x","chosen":"a","rejected":"b"}\n'
        '{"prompt":"x","chosen":"c","rejected":"d"}\n'
    )
    with pytest.raises(ValueError,match="duplicate"):
        validate_records(file,2)


def test_identical_chosen_rejected_rejected(tmp_path):
    file=tmp_path/"data.jsonl"
    file.write_text('{"prompt":"x","chosen":"same","rejected":"same"}\n')
    with pytest.raises(ValueError,match="malformed"):
        validate_records(file,1)
