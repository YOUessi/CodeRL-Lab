"""TRAIN-012: immutable CPU-only checkpoint parameter-shift audit.

Quantifies how much 7B QLoRA parameters changed between the fully trained
TRAIN-011 SFT adapter and TRAIN-012 DPO adapter. Does not read preference
labels or run a model, and avoids reporting a weight hash difference as
evidence of a meaningful functional shift.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

SFT_SHA = "25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8"
DPO_SHA = "d586d9ef32377189aa7d79b70ea5e37b5d5f26b930a4f60266ca024213f1a185"


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def analyze(sft_path: Path, dpo_path: Path) -> dict:
    import torch
    from safetensors.torch import load_file
    for path, sha in ((sft_path,SFT_SHA),(dpo_path,DPO_SHA)):
        if digest(path)!=sha:
            raise ValueError(f"unexpected frozen adapter SHA at {path}")
    a=load_file(str(sft_path),device="cpu")
    b=load_file(str(dpo_path),device="cpu")
    if set(a)!=set(b):
        raise ValueError("SFT/DPO state_dict keys do not match")
    changed=total=0
    delta_sq=base_sq=0.0
    max_abs=0.0
    dtypes_a=Counter()
    dtypes_b=Counter()
    for key in sorted(a):
        left,right=a[key],b[key]
        if left.shape!=right.shape:
            raise ValueError(f"adapter tensor shape mismatch: {key}")
        dtypes_a[str(left.dtype)]+=1
        dtypes_b[str(right.dtype)]+=1
        x=left.float()
        y=right.float()
        diff=y-x
        total+=left.numel()
        changed+=int((diff!=0).count_nonzero().item())
        delta_sq+=float(diff.square().sum().item())
        base_sq+=float(x.square().sum().item())
        max_abs=max(max_abs,float(diff.abs().max().item()))
    if total==0 or base_sq<=0:
        raise ValueError("cannot compare missing/zero tensors")
    return {
        "experiment":"TRAIN-012-ADAPTER-SHIFT",
        "status":"completed",
        "sft_sha256":SFT_SHA,
        "dpo_sha256":DPO_SHA,
        "tensors_compared":len(a),
        "parameters_compared":total,
        "parameters_changed":changed,
        "changed_fraction":changed/total,
        "relative_l2_parameter_shift":math.sqrt(delta_sq/base_sq),
        "max_absolute_parameter_difference":max_abs,
        "sft_tensor_dtypes":dict(dtypes_a),
        "dpo_tensor_dtypes":dict(dtypes_b),
        "interpretation":"Small bfloat16-quantized adapter drift after 237-step DPO. Not evidence alone of a numeric optimization bug; requires controlled comparison with FP32 trainable adapters.",
    }


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--sft",required=True,type=Path)
    p.add_argument("--dpo",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    result=analyze(args.sft,args.dpo)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():
        if json.loads(args.output.read_text())!=result:
            raise FileExistsError("refusing to overwrite non-identical adapter audit")
    else:
        args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
