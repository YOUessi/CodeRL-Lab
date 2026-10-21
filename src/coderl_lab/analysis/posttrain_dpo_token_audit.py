"""Audit effective DPO train/validation rows after TRL keep_start filtering.

This is CPU-only and pinned to the same Qwen tokenizer, UltraFeedback
manifest, DPO max_length and truncation mode. No model parameter updates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_split(
    rows: list[dict[str, Any]],
    *,
    tokenizer,
    max_length: int,
) -> dict[str, Any]:
    if not rows or max_length < 8:
        raise ValueError("need non-empty split and valid max_length")
    prompt_lengths: list[int] = []
    dropped: list[int] = []
    # TRL DPOTrainer._prepare_dataset uses _tokenize(tokenizer, text=prompt)
    # and filters len(prompt_ids) < max_length for keep_start truncation.
    # It does not filter by chosen/rejected model correctness.
    for i,row in enumerate(rows):
        prompt = str(row["prompt"])
        if not prompt.strip() or not str(row["chosen"]).strip() or not str(row["rejected"]).strip():
            raise ValueError(f"malformed pair at row {i}")
        prompt_ids = tokenizer(text=prompt)["input_ids"]
        n=len(prompt_ids)
        prompt_lengths.append(n)
        if n>=max_length:
            dropped.append(i)
    kept=len(rows)-len(dropped)
    return {
        "raw_pairs":len(rows),
        "kept_pairs_after_keep_start_prompt_filter":kept,
        "dropped_pairs_due_to_prompt_filling_max_length":len(dropped),
        "dropped_sample_indices":dropped,
        "max_length":max_length,
        "min_prompt_tokens":min(prompt_lengths),
        "mean_prompt_tokens":mean(prompt_lengths),
        "median_prompt_tokens":median(prompt_lengths),
        "max_prompt_tokens":max(prompt_lengths),
        "kept_fraction":kept/len(rows),
    }


def audit(
    *,
    config: Path,
    manifest: Path,
    train: Path,
    heldout: Path,
    tokenizer,
) -> dict[str, Any]:
    cfg=yaml.safe_load(config.read_text(encoding="utf-8"))
    m=json.loads(manifest.read_text(encoding="utf-8"))
    if cfg.get("experiment",{}).get("id")!="TRAIN-007C":
        raise ValueError("wrong experiment config")
    data=cfg["data"]
    if any((
        m.get("track")!="A-posttraining",
        m.get("stage")!="dpo",
        m.get("full_scan") is not True,
        m.get("prompt_overlap")!=0,
        m.get("dataset_revision")!=data["revision"],
        m["train"]["rows"]!=data["expected_train_pairs"],
        m["validation"]["rows"]!=data["expected_validation_pairs"],
        sha256(train)!=m["train"]["sha256"],
        sha256(heldout)!=m["validation"]["sha256"],
        cfg["model"]["name_or_path"]!="Qwen/Qwen3-1.7B-Base",
        cfg["model"]["revision"]!="ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
    )):
        raise ValueError("frozen real DPO source/revision/manifest mismatch")
    if cfg["training"].get("truncation_mode","keep_start")!="keep_start":
        raise ValueError("this audit only matches TRL default keep_start")
    def read(path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_rows=read(train)
    heldout_rows=read(heldout)
    if len(train_rows)!=m["train"]["rows"] or len(heldout_rows)!=m["validation"]["rows"]:
        raise ValueError("DPO row count mismatch")
    train_a=audit_split(train_rows,tokenizer=tokenizer,max_length=int(cfg["training"]["max_length"]))
    heldout_a=audit_split(heldout_rows,tokenizer=tokenizer,max_length=int(cfg["training"]["max_length"]))
    update_steps=math.ceil(train_a["kept_pairs_after_keep_start_prompt_filter"]/
                            int(cfg["training"]["per_device_train_batch_size"])/
                            int(cfg["training"]["gradient_accumulation_steps"]))
    return {
        "experiment":"TRAIN-007C",
        "phase":"post_training_start_audit_of_frozen_preprocessing_rule",
        "source_revision":m["dataset_revision"],
        "train_sha256":m["train"]["sha256"],
        "heldout_sha256":m["validation"]["sha256"],
        "config_sha256":sha256(config),
        "tokenizer":"Qwen/Qwen3-1.7B-Base",
        "tokenizer_revision":cfg["model"]["revision"],
        "trl_truncation_mode":"keep_start",
        "train":train_a,
        "heldout":heldout_a,
        "expected_optimizer_steps_one_epoch":update_steps,
        "new_cuda_training":False,
        "claim_limit":"Counts input preprocessing, not preference correctness or final DPO accuracy.",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,default=Path("configs/posttrain_a_dpo_qwen3_1.7b.yaml"))
    p.add_argument("--manifest",type=Path,default=Path("data/generated/posttrain-h4-v1/dpo_manifest.json"))
    p.add_argument("--train",type=Path,default=Path("data/generated/posttrain-h4-v1/dpo_train.jsonl"))
    p.add_argument("--heldout",type=Path,default=Path("data/generated/posttrain-h4-v1/dpo_validation.jsonl"))
    p.add_argument("--output",type=Path,default=Path("artifacts/train007c/preprocessing_audit.json"))
    a=p.parse_args()
    from transformers import AutoTokenizer
    cfg=yaml.safe_load(a.config.read_text(encoding="utf-8"))
    tokenizer=AutoTokenizer.from_pretrained(
        cfg["model"]["name_or_path"],revision=cfg["model"]["revision"],trust_remote_code=True,
    )
    result=audit(config=a.config,manifest=a.manifest,train=a.train,
                 heldout=a.heldout,tokenizer=tokenizer)
    if a.output.exists():
        raise FileExistsError("preprocessing audit already exists")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
