"""Preflight that differs ONLY in the trainable LoRA tensor dtype.

No GPU, no tests/answers used; checkpoint fingerprints match TRAIN-012.
"""
from __future__ import annotations
import argparse
import json
from copy import deepcopy
from pathlib import Path

import yaml
from coderl_lab.train.train012_preflight import (
    digest, validate, validate_records,
)


def normalized_matched_config(cfg: dict) -> dict:
    copy=deepcopy(cfg)
    copy["experiment"]["name"]="__MATCHED_EXPERIMENT__"
    copy["training"]["output_dir"]="__MATCHED_OUTPUT__"
    copy["training"]["trainable_adapter_precision"]="__ONLY_VARIABLE__"
    return copy


def compare_configs(bf16: dict, fp32: dict) -> None:
    if bf16["training"].get("trainable_adapter_precision")!="bf16":
        raise ValueError("BF16 arm must explicitly use trainable BF16 LoRA")
    if fp32["training"].get("trainable_adapter_precision")!="fp32":
        raise ValueError("FP32 arm must explicitly use trainable FP32 LoRA")
    if normalized_matched_config(bf16)!=normalized_matched_config(fp32):
        raise ValueError("experiment conditions differ in more than LoRA trainable dtype/output tag")
    for cfg in (bf16,fp32):
        if cfg["experiment"]["id"]!="TRAIN-013":
            raise ValueError("wrong experiment ID")
        if cfg["training"]["max_steps"]!=32:
            raise ValueError("formal protocol must have fixed32 optimizer steps")
        if cfg["training"]["learning_rate"]!=1e-6:
            raise ValueError("precision study must freeze TRAIN-012 learning rate")
        if cfg["training"]["save_strategy"]!="no":
            raise ValueError("matched32 must disable trainer checkpoint saving")
        if cfg["data"]["heldout_validation"] is not None:
            raise ValueError("heldout labels must never enter Trainer32")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--bf16",type=Path,required=True)
    p.add_argument("--fp32",type=Path,required=True)
    p.add_argument("--adapter",type=Path,required=True)
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--heldout",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    bf=yaml.safe_load(a.bf16.read_text())
    fp=yaml.safe_load(a.fp32.read_text())
    compare_configs(bf,fp)
    manifest=json.loads(a.manifest.read_text())
    sft=json.loads((a.adapter/"run_summary.json").read_text())
    train_prompts=validate_records(a.train,2048)
    val_prompts=validate_records(a.heldout,256)
    for cfg in (bf,fp):
        check=deepcopy(cfg)
        check["experiment"]["id"]="TRAIN-012"
        validate(
            cfg=check,train_manifest=manifest,sft_summary=sft,
            sft_sha=digest(a.adapter/"adapter_model.safetensors"),
            train_sha=digest(a.train),eval_sha=digest(a.heldout),
            train_prompts=train_prompts,eval_prompts=val_prompts,
        )
    output={
        "experiment":"TRAIN-013",
        "status":"frozen_paired_cpu_preflight_passed",
        "differing_training_variable":"trainable_adapter_precision",
        "bf16_config_sha256":digest(a.bf16),
        "fp32_config_sha256":digest(a.fp32),
        "SFT_adapter_sha256":digest(a.adapter/"adapter_model.safetensors"),
        "train_sha256":digest(a.train),
        "heldout_sha256":digest(a.heldout),
        "manifest_sha256":digest(a.manifest),
        "training_pairs":2048,"heldout_pairs":256,
        "train_heldout_prompt_overlap":0,
        "matched_optimizer_steps":32,"seed":42,
        "initializing_from_identical_7B_SFT":True,
        "gpu_train_completed":False,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():
        if json.loads(a.output.read_text())!=output:
            raise ValueError("existing freeze output drift")
    else:
        a.output.write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps(output,indent=2),flush=True)


if __name__=="__main__":main()
