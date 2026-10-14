from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from coderl_lab.train.train013_freeze import compare_configs
from coderl_lab.analysis.train013_heldout_preference import BETA, summarize, bootstrap


def configs():
    base=Path("configs")
    bf=yaml.safe_load((base/"train013_dpo_7b_matched32_bf16.yaml").read_text())
    fp=yaml.safe_load((base/"train013_dpo_7b_matched32_fp32.yaml").read_text())
    return bf,fp


def test_precision_is_the_only_train_config_variable():
    bf,fp=configs()
    compare_configs(bf,fp)
    bad=deepcopy(fp);bad["training"]["learning_rate"]*=10
    with pytest.raises(ValueError,match="more than"):
        compare_configs(bf,bad)
    bad=deepcopy(fp);bad["training"]["trainable_adapter_precision"]="bf16"
    with pytest.raises(ValueError,match="FP32 arm"):
        compare_configs(bf,bad)


def item(index:int, *, old:float, bf:float, fp:float):
    return {
        "index":index,"scorable":True,
        "sft_chosen_logp":old,"sft_rejected_logp":0.,
        "bf16_chosen_logp":bf,"bf16_rejected_logp":0.,
        "fp32_chosen_logp":fp,"fp32_rejected_logp":0.,
    }


def test_precision_matched_preference_summary():
    samples=[
        item(0,old=-1.,bf=-.5,fp=2.),
        item(1,old=2.,bf=1.,fp=-1.),
        item(2,old=-2.,bf=-1.2,fp=-1.1),
        {"index":3,"scorable":False},
    ]
    result=summarize(samples)
    assert result["total_pairs"]==4
    assert result["scorable_pairs"]==3
    assert result["unscorable_context512"]==1
    assert result["sft"]["raw_chosen_preferred"]==1
    assert result["bf16"]["raw_chosen_preferred"]==1
    assert result["fp32"]["raw_chosen_preferred"]==1
    assert result["fp32_minus_bf16"]["bf16_wrong_fp32_correct"]==1
    assert result["fp32_minus_bf16"]["bf16_correct_fp32_wrong"]==1
    assert result["bf16_minus_initial_sft"]["mean_implicit_reward_margin"]==pytest.approx(
        BETA*(.5 - 1 + .8)/3
    )
    assert result["fp32_minus_bf16"]["raw_chosen_preference_accuracy_delta_bootstrap"]["observed_mean"]==0.


def test_bootstrap_constant_is_exact():
    x=bootstrap([.7]*10,n=200,seed=42)
    assert x["observed_mean"]==pytest.approx(.7)
    assert x["ci95_low"]==pytest.approx(.7)
    assert x["ci95_high"]==pytest.approx(.7)
