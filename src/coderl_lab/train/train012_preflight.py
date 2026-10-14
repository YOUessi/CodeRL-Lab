"""TRAIN-012 CPU freeze gate for genuinely trained 7B SFT -> UltraFeedback DPO.

Verify original model, LoRA artifact, full-scan source and heldout splits
before any 7B GPU policy/reward preference update is permitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

MODEL = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
SFT_SHA = "25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8"
DPO_TRAIN_SHA = "945a336244462d43a7c3e70774158a4c40e873b5ee4f6c93d73e0d1e82222b77"
DPO_HELDOUT_SHA = "9d8a62dee49e9841add47a2ed27e485926c41e988ac06185ccd23b0df4e2e450"


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1048576),b""):
            h.update(chunk)
    return h.hexdigest()


def validate_records(path: Path, count: int) -> set[str]:
    prompts=set()
    found=0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item=json.loads(line)
            found+=1
            prompt=str(item.get("prompt",""))
            chosen=str(item.get("chosen",""))
            rejected=str(item.get("rejected",""))
            if (not prompt.strip() or not chosen.strip() or not rejected.strip()
                    or chosen==rejected or prompt in prompts):
                raise ValueError(f"malformed/duplicate DPO preference row #{found}")
            prompts.add(prompt)
    if found!=count:
        raise ValueError(f"expected {count} preference rows, got {found}")
    return prompts


def validate(
    *, cfg: dict, train_manifest: dict, sft_summary: dict,
    sft_sha: str, train_sha: str, eval_sha: str,
    train_prompts: set[str], eval_prompts: set[str],
) -> dict:
    if cfg["experiment"]["id"] != "TRAIN-012":
        raise ValueError("unexpected DPO training experiment")
    m=cfg["model"]
    if m["name_or_path"]!=MODEL or m["revision"]!=REVISION or m["dtype"]!="bfloat16":
        raise ValueError("7B pretrained weight mismatch")
    if cfg["quantization"]!={"mode":"nf4","double_quant":True}:
        raise ValueError("must match frozen SFT 4-bit quantization")
    if cfg["initial_policy"]["expected_sha256"] != SFT_SHA or sft_sha != SFT_SHA:
        raise ValueError("unverified source SFT adapter")
    if sft_summary.get("saved_adapter_sha256") != SFT_SHA:
        raise ValueError("TRAIN-011 SFT summary SHA mismatch")
    if sft_summary.get("model") != MODEL or sft_summary.get("requested_model_revision") != REVISION:
        raise ValueError("DPO source SFT base model mismatch")
    if sft_summary.get("num_examples") != 4096 or sft_summary.get("num_train_epochs") != 1:
        raise ValueError("source SFT checkpoint not full4096 formal training")
    if sft_summary.get("quantization") != {"mode":"nf4","double_quant":True,"compute_dtype":"bfloat16"}:
        raise ValueError("DPO/SFT precision mismatch")
    d=cfg["data"]
    if not (train_manifest.get("stage")=="dpo" and train_manifest.get("full_scan") is True
            and train_manifest.get("prompt_overlap")==0
            and train_manifest.get("dataset_revision")==d["revision"]):
        raise ValueError("DPO dataset must be original full-scan frozen source")
    if (
        train_manifest["train"]["rows"]!=2048
        or train_manifest["validation"]["rows"]!=256
        or len(train_prompts)!=2048 or len(eval_prompts)!=256
    ):
        raise ValueError("wrong split counts")
    if train_prompts & eval_prompts:
        raise ValueError("preference train/heldout prompt leakage")
    if (
        train_sha!=DPO_TRAIN_SHA or eval_sha!=DPO_HELDOUT_SHA
        or d["train_sha256"]!=train_sha or d["heldout_sha256"]!=eval_sha
        or train_manifest["train"]["sha256"]!=train_sha
        or train_manifest["validation"]["sha256"]!=eval_sha
    ):
        raise ValueError("training or validation data SHA drift")
    train=cfg["training"]
    if (train["seed"]!=42 or train["max_length"]!=512
            or train["gradient_accumulation_steps"]!=8
            or float(train["beta"])!=0.1
            or train["loss_type"]!="sigmoid"):
        raise ValueError("DPO pre-registration math/length settings changed")
    return {
        "experiment":"TRAIN-012",
        "status":"cpu_preflight_passed",
        "sft_adapter_sha256":sft_sha,
        "source_model":MODEL,
        "model_revision":REVISION,
        "ultrafeedback_train_sha256":train_sha,
        "ultrafeedback_heldout_sha256":eval_sha,
        "train_pairs":len(train_prompts),
        "heldout_pairs":len(eval_prompts),
        "prompts_disjoint":True,
        "validation_never_used_as_optimizer_data":True,
        "new_7b_dpo_gpu_training_completed":False,
    }


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--sft-adapter",type=Path,required=True)
    p.add_argument("--dpo-train",type=Path,required=True)
    p.add_argument("--dpo-validation",type=Path,required=True)
    p.add_argument("--dpo-manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    cfg=yaml.safe_load(a.config.read_text(encoding="utf-8"))
    manifest=json.loads(a.dpo_manifest.read_text(encoding="utf-8"))
    sft_summary=json.loads((a.sft_adapter/"run_summary.json").read_text(encoding="utf-8"))
    output=validate(
        cfg=cfg,train_manifest=manifest,sft_summary=sft_summary,
        sft_sha=digest(a.sft_adapter/"adapter_model.safetensors"),
        train_sha=digest(a.dpo_train),eval_sha=digest(a.dpo_validation),
        train_prompts=validate_records(a.dpo_train,2048),
        eval_prompts=validate_records(a.dpo_validation,256),
    )
    output["config_sha256"]=digest(a.config)
    output["manifest_sha256"]=digest(a.dpo_manifest)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():
        if json.loads(a.output.read_text(encoding="utf-8"))!=output:
            raise FileExistsError("preflight identity mismatch; not overwriting")
    else:
        a.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(output,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
