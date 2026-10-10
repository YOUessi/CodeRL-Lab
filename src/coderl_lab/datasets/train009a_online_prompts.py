"""TRAIN-009A: fixed, prompt-only RLHF pool from disjoint UltraChat prompts.

No label/answer is exposed to GRPO as a training target. UltraFeedback was
already used to train the reward model, so its train AND validation prompts
are excluded from online policy prompts. Only frozen SHA-pinned input files.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ULTRACHAT_REV = "b7fe606ecdbf71e8537946a8d9de5ccf0f6da48b"
ULTRAFEEDBACK_REV = "daee4cffe8bea93b1cd7874b01c5a6a4ae9dcaf2"
SFT_TRAIN_SHA = "f868096a21eb37249d06d318fb56ab3b0e3e99c4db442e54c43d6f42b565888b"
SFT_VALID_SHA = "a7f6e07f8569157fe5c3ae0deb65875c3427f9db3d6a5120379e1d9b10dae170"
RM_TRAIN_SHA = "945a336244462d43a7c3e70774158a4c40e873b5ee4f6c93d73e0d1e82222b77"
RM_VALID_SHA = "9d8a62dee49e9841add47a2ed27e485926c41e988ac06185ccd23b0df4e2e450"


def digest_file(path: Path) -> str:
    sha=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):
            sha.update(block)
    return sha.hexdigest()


def digest_prompt(prompt:str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def read_rows(path:Path) -> list[dict[str,Any]]:
    rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty RLHF dataset: {path}")
    return rows


def frozen_prompt_pool(*, root:Path, seed:int=42,
                       max_train_prompts:int|None=None)->tuple[list[dict[str,str]],dict[str,Any]]:
    if max_train_prompts is not None and max_train_prompts<1:
        raise ValueError("max_train_prompts must be >= 1")
    expected={
        "sft_train.jsonl":SFT_TRAIN_SHA,"sft_validation.jsonl":SFT_VALID_SHA,
        "dpo_train.jsonl":RM_TRAIN_SHA,"dpo_validation.jsonl":RM_VALID_SHA,
    }
    for filename,digest in expected.items():
        if digest_file(root/filename)!=digest:
            raise ValueError(f"TRAIN-009A input SHA mismatch: {filename}")
    sft_manifest=json.loads((root/"sft_manifest.json").read_text(encoding="utf-8"))
    dpo_manifest=json.loads((root/"dpo_manifest.json").read_text(encoding="utf-8"))
    for manifest,stage,revision,counts,hashes in (
        (sft_manifest,"sft",ULTRACHAT_REV,(4096,256),(SFT_TRAIN_SHA,SFT_VALID_SHA)),
        (dpo_manifest,"dpo",ULTRAFEEDBACK_REV,(2048,256),(RM_TRAIN_SHA,RM_VALID_SHA)),
    ):
        if any((manifest.get("stage")!=stage,
                manifest.get("track")!="A-posttraining",
                manifest.get("full_scan") is not True,
                manifest.get("dataset_revision")!=revision,
                manifest.get("prompt_overlap")!=0,
                manifest.get("train",{}).get("rows")!=counts[0],
                manifest.get("validation",{}).get("rows")!=counts[1],
                manifest.get("train",{}).get("sha256")!=hashes[0],
                manifest.get("validation",{}).get("sha256")!=hashes[1])):
            raise ValueError(f"TRAIN-009A frozen {stage} manifest invalid")
    sft_train=read_rows(root/"sft_train.jsonl")
    sft_eval=read_rows(root/"sft_validation.jsonl")
    rm_train=read_rows(root/"dpo_train.jsonl")
    rm_eval=read_rows(root/"dpo_validation.jsonl")
    if tuple(map(len,(sft_train,sft_eval,rm_train,rm_eval)))!=(4096,256,2048,256):
        raise ValueError("frozen RLHF source row count mismatch")
    def extract(rows:list[dict[str,Any]],name:str) -> set[str]:
        out=set()
        for i,row in enumerate(rows):
            p=row.get("prompt")
            if not isinstance(p,str) or not p.strip():
                raise ValueError(f"invalid {name} prompt {i}")
            out.add(p)
        if len(out)!=len(rows):
            raise ValueError(f"duplicate {name} prompts")
        return out
    sft_eval_prompts=extract(sft_eval,"SFT validation")
    rm_protected=extract(rm_train,"RM train")|extract(rm_eval,"RM validation")
    rm_protected.add("")  # never admit an empty prompt
    train_prompts=extract(sft_train,"SFT train")
    if train_prompts&sft_eval_prompts:
        raise ValueError("SFT train/heldout overlaps")
    eligible=[(index,row["prompt"]) for index,row in enumerate(sft_train)
              if row["prompt"] not in rm_protected and row["prompt"] not in sft_eval_prompts]
    count=len(eligible)
    if count<4 or (max_train_prompts is not None and count<max_train_prompts):
        raise ValueError("insufficient disjoint RLHF training prompts")
    eligible.sort(key=lambda pair:(
        hashlib.sha256(f"{seed}|009A|{pair[1]}".encode("utf-8")).digest(),pair[0]))
    if max_train_prompts is not None:
        eligible=eligible[:max_train_prompts]
    chosen=[{"prompt":prompt,"prompt_sha256":digest_prompt(prompt)} for _,prompt in eligible]
    # We do not pass SFT completions into the online actor's dataset.
    fingerprint=hashlib.sha256(("\n".join(x["prompt_sha256"] for x in chosen)+"\n").encode()).hexdigest()
    audit={
        "experiment":"TRAIN-009A",
        "source":"UltraChat SFT train 4096, prompt-only, RM trained on disjoint UltraFeedback train/heldout",
        "seed":seed,
        "sft_source_revision":ULTRACHAT_REV,
        "reward_source_revision":ULTRAFEEDBACK_REV,
        "input_sha256":expected,
        "sft_train_raw_prompts":4096,
        "sft_valid_raw_prompts":256,
        "reward_train_prompts":2048,
        "reward_validation_prompts":256,
        "discarded_train_due_to_rm_any_split_overlap":4096-count,
        "eligible_train_prompts":count,
        "selected_train_prompts":len(chosen),
        "selected_prompt_hashes_sha256":fingerprint,
        "contains_sft_response_targets":False,
        "contains_reward_train_or_validation_prompts":False,
        "heldout_SFT_validation_excluded":True,
        "contains_private_hidden_test_data":False,
    }
    return chosen,audit
