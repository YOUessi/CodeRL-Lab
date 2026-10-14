"""TRAIN-013 full237 paired BF16/FP32 parameter + training dynamic audit.

Existing TRAIN-012 BF16 237-step checkpoint is immutable. Compare with the
single-variable FP32 237-step TRAIN-013 run; no preference labels are read.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import yaml

from coderl_lab.train.train012_preflight import digest
from coderl_lab.train.train013_full237_freeze import verify_difference
from coderl_lab.analysis.train013_precision_report import (
    compare_adapters, load_step_metrics,
)


def verify_run(old:dict,new:dict,freeze:dict, old_path:Path, new_path:Path):
    if (freeze["status"]!="frozen_pre_gpu"
        or freeze["old_training_steps"]!=237
        or freeze["new_target_steps"]!=237):
        raise ValueError("full237 controlled study was not frozen first")
    if digest(old_path)!=freeze["prior_full237_run_summary_sha256"]:
        raise ValueError("prior old BF16 DPO summary changed")
    for run in (old,new):
        if (run["global_step"]!=237 or run["trainer_max_steps"]!=237
            or run["final_epoch"]!=1.0
            or run["checkpoint_policy"]["identity"]["training_data_sha256"]!=freeze["training_data_sha256"]
            or run["sft_adapter_sha256"]!=freeze["SFT_adapter_sha256"]
            or run["num_pairs"]!=2048):
            raise ValueError("not matched complete full237 DPO")
    if old.get("trainable_adapter_precision") is not None:
        raise ValueError("TRAIN-012 baseline unexpectedly changed from TRL BF16 default")
    if new.get("trainable_adapter_precision",{}).get("trainable_dtypes_after")!=["torch.float32"]:
        raise ValueError("FP32 full237 not actually trained with FP32 LoRA parameters")
    if new["trainable_adapter_precision"]["paired_ref_initially_identical"] is not True:
        raise ValueError("FP32 full237 reference not identical initially")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--sft",type=Path,required=True)
    p.add_argument("--prior-bf16",type=Path,required=True)
    p.add_argument("--new-fp32",type=Path,required=True)
    p.add_argument("--prior-config",type=Path,required=True)
    p.add_argument("--new-config",type=Path,required=True)
    p.add_argument("--freeze",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    prior_cfg=yaml.safe_load(a.prior_config.read_text())
    current_cfg=yaml.safe_load(a.new_config.read_text())
    verify_difference(prior_cfg,current_cfg)
    frozen=json.loads(a.freeze.read_text())
    if (digest(a.prior_config)!=frozen["old_training_config_sha256"]
        or digest(a.new_config)!=frozen["new_training_config_sha256"]):
        raise ValueError("full237 config SHA mismatch")
    old_path=a.prior_bf16/"run_summary.json"
    new_path=a.new_fp32/"run_summary.json"
    old=json.loads(old_path.read_text())
    new=json.loads(new_path.read_text())
    verify_run(old,new,frozen,old_path,new_path)
    histories={
        "bf16":load_step_metrics(a.prior_bf16,steps=237),
        "fp32":load_step_metrics(a.new_fp32,steps=237),
    }
    if any(
        x.get("num_tokens")!=y.get("num_tokens")
        for x,y in zip(histories["bf16"],histories["fp32"])
    ):
        raise ValueError("train token counts not aligned across 237 steps")
    for arm,folder,summary in (("bf16",a.prior_bf16,old),("fp32",a.new_fp32,new)):
        if digest(folder/"adapter_model.safetensors")!=summary["output_adapter_sha256"]:
            raise ValueError(f"{arm} full237 adapter SHA mismatch")
    diff=compare_adapters(
        sft=a.sft,
        bf16=a.prior_bf16/"adapter_model.safetensors",
        fp32=a.new_fp32/"adapter_model.safetensors",
    )
    per_arm={}
    for arm,record in (("bf16",old),("fp32",new)):
        rows=histories[arm]
        per_arm[arm]={
            "adapter_sha256":record["output_adapter_sha256"],
            "training_loss":record["metrics"]["train_loss"],
            "trainer_runtime_seconds":record["metrics"]["train_runtime"],
            "last_loss":rows[-1]["loss"],
            "mean_train_reward_margin":statistics.mean(float(r.get("rewards/margins",0.)) for r in rows),
            "mean_train_reward_accuracy":statistics.mean(float(r.get("rewards/accuracies",0.)) for r in rows),
            "last_reward_margin":rows[-1].get("rewards/margins"),
            "GPU_peak_reserved_mib":record["gpu"]["peak_reserved_bytes"]/2**20,
        }
    result={
        "experiment":"TRAIN-013-FULL237-PRECISION-AUDIT",
        "status":"completed",
        "only_changed_trainable_model_property":"BF16 -> FP32 LoRA trainable storage",
        "pre_registered_precision_freeze_sha256":digest(a.freeze),
        "matched_optimizer_steps":237,
        "exactly_matching_tokens_each_step":True,
        "same_frozen_initial_SFT_adapter":True,
        "reference_policy_frozen":True,
        "per_arm":per_arm,
        "parameter_comparison":diff,
        "limitations":"Parameter distance and training rewards do not establish new independent preference accuracy. The existing heldout set was inspected before.",
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    if a.out.exists():
        if json.loads(a.out.read_text())!=result:
            raise FileExistsError("refuse to replace different 237-step audit")
    else:
        a.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
