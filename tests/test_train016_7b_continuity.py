"""TRAIN-016: ensure 7B continuity, quantized GRPO and SFT baseline contract."""
import json
from pathlib import Path

import yaml

from coderl_lab.train.train016_7b_qlora_grpo import MODEL, REVISION, FROZEN_SFT_SHA
from coderl_lab.analysis.train016_7b_sft_grpo_holdout import (
    ARMS, SOURCE_SFT_SHA, TASKS, SAMPLES, sample_seed,
)


def test_7b_backbone_not_accidentally_the_15b_engineering_probe():
    cfg=yaml.safe_load(Path("configs/train016_7b_qlora_grpo.yaml").read_text())
    assert MODEL=="Qwen/Qwen2.5-7B-Instruct"
    assert cfg["model"]["name_or_path"]==MODEL
    assert cfg["model"]["revision"]==REVISION
    assert cfg["experiment"]["id"]=="TRAIN-016"
    assert cfg["grpo"]["init_from_sft_adapter"] is True
    assert cfg["grpo"]["adapter_sha256"]==FROZEN_SFT_SHA
    assert cfg["grpo"]["qlora_base_quantization"]=="nf4_double_quant_bf16"
    assert cfg["grpo"]["adapter_trainable_dtype_after_trainer"]=="float32"
    assert cfg["grpo"]["num_generations"]==2
    assert cfg["grpo"]["per_device_train_batch_size"]==2


def test_posttraining_uses_exact_source_7b_sft_model():
    assert SOURCE_SFT_SHA==FROZEN_SFT_SHA
    assert ARMS==("sft","grpo")
    assert TASKS==12 and SAMPLES==4
    assert sample_seed("heldout/pairs/8",0)==sample_seed("heldout/pairs/8",0)
    assert sample_seed("heldout/pairs/8",1)!=sample_seed("heldout/pairs/8",0)


def test_reward_dataset_is_frozen_old_algorithmic_exploration_not_new_blind():
    cfg=yaml.safe_load(Path("configs/train016_7b_qlora_grpo.yaml").read_text())
    assert cfg["dataset"]["train_sha256"]=="c076b6a3d685ed1b3b4df20e944be78064f069371a1ef177ed8b2e6342eabfe7"
    assert cfg["dataset"]["heldout_public_sha256"]=="009dc276e718d4ec600faa3b885f0f934c225d0dd0637d6d8d3b468caaa6d834"
    assert cfg["dataset"]["heldout_oracle_sha256"]=="cfcd2467b8c43fa084a63f7fcf380fe58bd6d483b2530dbbab358b10cd6a822e"
    assert cfg["runs"]["smoke_output"]!=cfg["runs"]["formal_output"]
