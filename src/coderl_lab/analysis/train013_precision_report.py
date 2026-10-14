"""TRAIN-013 paired LoRA precision / loss dynamics audit.

Only BF16/FP32 storage of the trainable default adapter is varied.
All checkpoints, config hashes, reference initialization and step counts
must match preregistered provenance. This code never reads heldout labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from coderl_lab.train.train012_preflight import digest
from coderl_lab.train.train013_freeze import compare_configs


def verify_summary(data: dict, expected: str, *, steps: int) -> None:
    precision=data.get("trainable_adapter_precision", {})
    if (
        data.get("global_step") != steps
        or data.get("trainer_max_steps") != steps
        or data.get("sft_adapter_sha256") != "25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8"
        or precision.get("requested") != expected
        or precision.get("trainable_parameters") != 2523136
        or precision.get("trainable_tensors") != 112
        or precision.get("frozen_reference_tensors") != 112
        or precision.get("paired_ref_initially_identical") is not True
        or precision.get("trainable_dtypes_after") != (["torch.bfloat16"] if expected == "bf16" else ["torch.float32"])
        or precision.get("frozen_reference_dtypes") != ["torch.float32"]
    ):
        raise ValueError("TRAIN-013 unmatched/incomplete real GPU optimization")
    if data.get("num_heldout_preference_pairs") != 0:
        raise ValueError("heldout scoring accidentally used during DPO training")


def load_step_metrics(folder: Path, steps: int=32) -> list[dict]:
    rows=json.loads((folder/"log_history.json").read_text(encoding="utf-8"))
    updates=[x for x in rows if isinstance(x.get("step"),int) and "loss" in x]
    if len(updates)!=steps or [x["step"] for x in updates] != list(range(1,steps+1)):
        raise ValueError("not exactly matched train steps in training log")
    return updates


def compare_adapters(*, sft: Path, bf16: Path, fp32: Path) -> dict:
    import torch
    from safetensors.torch import load_file

    expected_sft="25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8"
    if digest(sft)!=expected_sft:
        raise ValueError("source SFT adapter is not frozen")
    a=load_file(str(sft),device="cpu")
    b=load_file(str(bf16),device="cpu")
    c=load_file(str(fp32),device="cpu")
    if not set(a)==set(b)==set(c) or len(a)!=112:
        raise ValueError("adapter tensors do not match expected identical LoRA structure")
    if {str(x.dtype) for x in a.values()} != {"torch.bfloat16"}:
        raise ValueError("original SFT LoRA must be BF16")
    if {str(x.dtype) for x in b.values()} != {"torch.bfloat16"}:
        raise ValueError("BF16 arm output is not BF16")
    if {str(x.dtype) for x in c.values()} != {"torch.float32"}:
        raise ValueError("FP32 arm output did not retain FP32 dtype")
    count=0
    changed_b=changed_c=changed_c_bf16=0
    sum_a_sq=delta_b_sq=delta_c_sq=delta_bc_sq=0.0
    max_b=max_c=0.0
    for key in sorted(a):
        x=a[key]
        y=b[key]
        z=c[key]
        if x.shape!=y.shape or x.shape!=z.shape:
            raise ValueError("LoRA tensor shape differs")
        x32=x.float()
        dy=y.float()-x32
        dz=z.float()-x32
        diff=y.float()-z.float()
        count+=x.numel()
        changed_b+=int(torch.count_nonzero(y!=x))
        changed_c+=int(torch.count_nonzero(z!=x32))
        changed_c_bf16+=int(torch.count_nonzero(z.to(torch.bfloat16)!=x))
        sum_a_sq+=float(x32.square().sum())
        delta_b_sq+=float(dy.square().sum())
        delta_c_sq+=float(dz.square().sum())
        delta_bc_sq+=float(diff.square().sum())
        max_b=max(max_b,float(dy.abs().max()))
        max_c=max(max_c,float(dz.abs().max()))
    if count!=2523136 or sum_a_sq==0.0:
        raise ValueError("unexpected trained parameters")
    return {
        "tensors":112,
        "compared_trainable_parameters":count,
        "bf16_parameters_changed":changed_b,
        "bf16_changed_fraction":changed_b/count,
        "fp32_parameters_changed":changed_c,
        "fp32_changed_fraction":changed_c/count,
        "fp32_parameters_changed_after_rounding_to_bf16":changed_c_bf16,
        "fp32_changed_after_rounding_fraction":changed_c_bf16/count,
        "bf16_relative_l2_shift":math.sqrt(delta_b_sq/sum_a_sq),
        "fp32_relative_l2_shift":math.sqrt(delta_c_sq/sum_a_sq),
        "bf16_vs_fp32_relative_l2_distance":math.sqrt(delta_bc_sq/sum_a_sq),
        "bf16_max_abs_weight_shift":max_b,
        "fp32_max_abs_weight_shift":max_c,
    }


def analysis(*, bf16_dir:Path, fp32_dir:Path, sft_path:Path, bf_cfg:Path, fp_cfg:Path, cpu_freeze:Path) -> dict:
    import yaml
    a=yaml.safe_load(bf_cfg.read_text())
    b=yaml.safe_load(fp_cfg.read_text())
    compare_configs(a,b)
    frozen=json.loads(cpu_freeze.read_text())
    if (
        frozen.get("status")!="frozen_paired_cpu_preflight_passed"
        or frozen.get("bf16_config_sha256")!=digest(bf_cfg)
        or frozen.get("fp32_config_sha256")!=digest(fp_cfg)
    ):
        raise ValueError("pre-registered experiment/data contract drift")
    left=json.loads((bf16_dir/"run_summary.json").read_text())
    right=json.loads((fp32_dir/"run_summary.json").read_text())
    verify_summary(left,"bf16",steps=32)
    verify_summary(right,"fp32",steps=32)
    for cfg,summary in ((bf_cfg,left),(fp_cfg,right)):
        if summary["checkpoint_policy"]["identity"]["config_sha256"]!=digest(cfg):
            raise ValueError("training code used different frozen config")
        if summary["checkpoint_policy"]["identity"]["training_data_sha256"]!=frozen["train_sha256"]:
            raise ValueError("training used different frozen 2048 source")
    histories={
        "bf16":load_step_metrics(bf16_dir),
        "fp32":load_step_metrics(fp32_dir),
    }
    for l,r in zip(histories["bf16"],histories["fp32"]):
        if l["step"]!=r["step"] or l.get("num_tokens")!=r.get("num_tokens"):
            raise ValueError("optimizer step/data order mismatch between precision arms")
    outputs = {
        "bf16":bf16_dir/"adapter_model.safetensors",
        "fp32":fp32_dir/"adapter_model.safetensors",
    }
    if any(digest(outputs[arm])!=s["output_adapter_sha256"] for arm,s in (("bf16",left),("fp32",right))):
        raise ValueError("final saved adapter mismatch with TRAIN-013 trainer summary")
    weights=compare_adapters(sft=sft_path,bf16=outputs["bf16"],fp32=outputs["fp32"])
    metrics={}
    for arm,s in (("bf16",left),("fp32",right)):
        rows=histories[arm]
        metrics[arm]={
            "training_loss":s["metrics"]["train_loss"],
            "training_runtime_seconds":s["metrics"]["train_runtime"],
            "GPU_peak_allocated_mib":s["gpu"]["peak_allocated_bytes"]/2**20,
            "GPU_peak_reserved_mib":s["gpu"]["peak_reserved_bytes"]/2**20,
            "source_adapter_sha256":s["sft_adapter_sha256"],
            "output_adapter_sha256":s["output_adapter_sha256"],
            "adapter_tensor_dtypes":s["trainable_adapter_precision"],
            "loss_at_step_1":rows[0]["loss"],
            "loss_at_step_32":rows[-1]["loss"],
            "mean_train_reward_accuracy":statistics.mean(float(x.get("rewards/accuracies",0.)) for x in rows),
            "mean_train_reward_margin":statistics.mean(float(x.get("rewards/margins",0.)) for x in rows),
            "last_train_reward_margin":rows[-1].get("rewards/margins"),
            "mean_train_gradient_norm":statistics.mean(float(x["grad_norm"]) for x in rows),
        }
    return {
        "experiment":"TRAIN-013-MATCHED32-BF16-FP32",
        "status":"completed",
        "claimed_scope":"32-step precision mechanics control only; not new blind heldout effectiveness",
        "train_pairs_source":2048,
        "optimizer_steps_each":32,
        "sft_weights_identical_before_training":True,
        "identical_train_token_counts_each_step":True,
        "same_reference_policy_frozen":True,
        "only_changed_variable":"trainable_default_LoRA_parameter_storage_dtype",
        "source_config_sha256":{arm:digest(p) for arm,p in (("bf16",bf_cfg),("fp32",fp_cfg))},
        "metrics":metrics,
        "adapter_parameter_delta":weights,
        "limitations":"BF16/FP32 parameter changes alone are not evidence of better preference following or out-of-distribution capabilities; any heldout metrics seen previously are exploratory.",
    }


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--bf16-dir",required=True,type=Path)
    p.add_argument("--fp32-dir",required=True,type=Path)
    p.add_argument("--sft",required=True,type=Path)
    p.add_argument("--bf16-config",required=True,type=Path)
    p.add_argument("--fp32-config",required=True,type=Path)
    p.add_argument("--freeze",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    result=analysis(
        bf16_dir=a.bf16_dir,fp32_dir=a.fp32_dir,sft_path=a.sft,
        bf_cfg=a.bf16_config,fp_cfg=a.fp32_config,cpu_freeze=a.freeze,
    )
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():
        if json.loads(a.output.read_text())!=result:
            raise FileExistsError("existing precision report differs")
    else:
        a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
