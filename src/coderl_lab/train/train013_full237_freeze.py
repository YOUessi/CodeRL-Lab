"""Freeze exact full-epoch TRAIN-012 BF16 vs TRAIN-013 FP32-only difference.

Executed on Tang *before* starting the 237-step FP32 full run; makes no
heldout quality decisions. The previously completed TRAIN-012 config is
the comparator, NOT a rerun or overwritten artifact.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import yaml

from coderl_lab.train.train012_preflight import validate,validate_records,digest


def normalized(config: dict) -> dict:
    d=deepcopy(config)
    d["experiment"]["id"]="__EXPERIMENT_ID__"
    d["experiment"]["name"]="__EXPERIMENT_NAME__"
    d["training"]["output_dir"]="__SEPARATE_RESULT_PATH__"
    d["training"].pop("trainable_adapter_precision",None)
    return d


def verify_difference(old:dict, new:dict):
    if old["experiment"]["id"]!="TRAIN-012" or new["experiment"]["id"]!="TRAIN-013":
        raise ValueError("wrong old/new experiment IDs")
    if old["training"].get("trainable_adapter_precision") is not None:
        raise ValueError("baseline must use original TRL forced BF16 behavior")
    if new["training"].get("trainable_adapter_precision")!="fp32":
        raise ValueError("new arm must explicitly force FP32 trainable LoRA")
    if normalized(old)!=normalized(new):
        raise ValueError("matched full237 changed fields other than adapter dtype/metadata")
    if new["training"].get("max_steps")!=-1:
        raise ValueError("full237 should preserve 1 epoch with auto epoch length")
    if new["training"].get("save_strategy")!="steps":
        raise ValueError("must keep original checkpoint policy")
    if new["training"].get("save_steps")!=32:
        raise ValueError("save every32 optimizer steps exactly as TRAIN-012")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--prior-config",type=Path,required=True)
    ap.add_argument("--new-config",type=Path,required=True)
    ap.add_argument("--prior-summary",type=Path,required=True)
    ap.add_argument("--sft-adapter",type=Path,required=True)
    ap.add_argument("--train",type=Path,required=True)
    ap.add_argument("--heldout",type=Path,required=True)
    ap.add_argument("--manifest",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    prior=yaml.safe_load(a.prior_config.read_text())
    current=yaml.safe_load(a.new_config.read_text())
    verify_difference(prior,current)
    previous=json.loads(a.prior_summary.read_text())
    if (
        previous.get("global_step")!=237 or previous.get("trainer_max_steps")!=237
        or previous.get("final_epoch")!=1.0
        or previous.get("quantization",{}).get("mode")!="nf4"
        or previous.get("trainable_adapter_precision",{}).get("requested") is not None
    ):
        # TRAIN-012 finished before the adapter dtype option existed.
        raise ValueError("wrong previous completed BF16 DPO training record")
    sft=json.loads((a.sft_adapter/"run_summary.json").read_text())
    manifest=json.loads(a.manifest.read_text())
    clone=deepcopy(current)
    clone["experiment"]["id"]="TRAIN-012"
    validate(
        cfg=clone,train_manifest=manifest,sft_summary=sft,
        sft_sha=digest(a.sft_adapter/"adapter_model.safetensors"),
        train_sha=digest(a.train),eval_sha=digest(a.heldout),
        train_prompts=validate_records(a.train,2048),
        eval_prompts=validate_records(a.heldout,256),
    )
    result={
        "experiment":"TRAIN-013-FP32-FULL237-ONLY-PRECISION",
        "status":"frozen_pre_gpu",
        "old_training_config_sha256":digest(a.prior_config),
        "new_training_config_sha256":digest(a.new_config),
        "prior_full237_run_summary_sha256":digest(a.prior_summary),
        "SFT_adapter_sha256":digest(a.sft_adapter/"adapter_model.safetensors"),
        "training_data_sha256":digest(a.train),
        "heldout_data_sha256":digest(a.heldout),
        "manifest_sha256":digest(a.manifest),
        "old_training_steps":237,
        "new_target_steps":237,
        "only_changed_trainable_property":"LoRA default BF16 versus LoRA default FP32; identical frozen reference BF16 study values",
        "data_untouched":True,
        "new_training_completed":False,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():
        if json.loads(a.output.read_text())!=result:
            raise FileExistsError("frozen full237 change-detection mismatch")
    else:
        a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
