"""TRAIN-009B: audited actual SFT EOS/length-cap behavior on frozen prompts.

This is generation-only. It never trains, reads hidden tests, or uploads
prompts/completions. Prefix cap 96/256/512 is measured from a single seeded
max-512 rollout to avoid cherry-picked reruns at individual budgets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any

from coderl_lab.datasets.train009a_online_prompts import (
    SFT_VALID_SHA, read_rows, digest_prompt, frozen_prompt_pool,
)
from coderl_lab.train.learned_reward_grpo import (
    MODEL, REVISION, SFT_SHA, sha256,
)

BUDGETS=(96,256,512)
SAMPLE_SEED=42
EXPECTED_MODEL_EOS=151643
ADDITIONAL_IM_END=151645


def frozen_holdout(root:Path, *, count:int=16,seed:int=SAMPLE_SEED)->list[dict[str,str]]:
    if count<1:
        raise ValueError("heldout selection count must be positive")
    train, _ = frozen_prompt_pool(root=root,seed=seed,max_train_prompts=16)
    sft_train={r["prompt"] for r in read_rows(root/"sft_train.jsonl")}
    reward_seen={r["prompt"] for path in ("dpo_train.jsonl","dpo_validation.jsonl")
                 for r in read_rows(root/path)}
    heldout=read_rows(root/"sft_validation.jsonl")
    if sha256(root/"sft_validation.jsonl")!=SFT_VALID_SHA or len(heldout)!=256:
        raise ValueError("frozen SFT test snapshot SHA or row count drift")
    if not all(isinstance(r.get("prompt"),str) and r["prompt"].strip() for r in heldout):
        raise ValueError("invalid heldout prompt")
    if len({r["prompt"] for r in heldout})!=len(heldout):
        raise ValueError("duplicate SFT heldout prompt")
    eligible=[r["prompt"] for r in heldout
              if r["prompt"] not in sft_train and r["prompt"] not in reward_seen]
    if len(eligible)<count:
        raise ValueError("insufficient reward-unseen and SFT-train-unseen prompts")
    eligible.sort(key=lambda text:hashlib.sha256(
        f"{seed}|009B|heldout|{text}".encode("utf-8")).digest())
    return [{"prompt":s,"prompt_sha256":digest_prompt(s)} for s in eligible[:count]]


def boundary_stats(
    token_ids:list[int], *, eos_id:int=EXPECTED_MODEL_EOS,
    im_end_id:int=ADDITIONAL_IM_END,
    budgets:tuple[int,...]=BUDGETS,
)->dict[str,Any]:
    if (not token_ids or not budgets or max(budgets)>512 or min(budgets)<1
            or len(set(budgets))!=len(budgets)):
        raise ValueError("invalid cap or empty decoder trace")
    eos_pos=next((i+1 for i,t in enumerate(token_ids) if t==eos_id),None)
    end_turn_pos=next((i+1 for i,t in enumerate(token_ids) if t==im_end_id),None)
    if eos_pos is not None and eos_pos!=len(token_ids):
        raise ValueError("tokens were generated after forced EOS stop")
    result={
        "generated_tokens":len(token_ids),
        "default_eos_position":eos_pos,
        "im_end_position":end_turn_pos,
        "default_eos_seen":eos_pos is not None,
        "im_end_seen":end_turn_pos is not None,
        "budgets":{},
    }
    for budget in budgets:
        # If the run terminated naturally before budget, it would also
        # terminate under that larger/equal max cap with identical RNG.
        natural=eos_pos is not None and eos_pos<=budget
        result["budgets"][str(budget)]={
            "default_eos_terminated":natural,
            "would_hit_cap":not natural,
            "completion_tokens":min(budget,eos_pos or budget),
            "im_end_seen_by_cap":end_turn_pos is not None and end_turn_pos<=budget,
        }
    return result


def summarize(rows:list[dict[str,Any]],*, budgets:tuple[int,...]=BUDGETS)->dict[str,Any]:
    if len(rows)!=32:
        raise ValueError("full 16 train/16 heldout fixed sample required")
    if len({(r["split"],r["prompt_sha256"]) for r in rows})!=32:
        raise ValueError("duplicate independent trace input")
    if {r["split"] for r in rows}!={"train_smoke","sft_heldout"}:
        raise ValueError("both frozen cohorts required")
    if any(sum(r["split"]==split for r in rows)!=16 for split in ("train_smoke","sft_heldout")):
        raise ValueError("16 rows from each cohort required")
    by={}
    for split in ("train_smoke","sft_heldout"):
        cohort=[r for r in rows if r["split"]==split]
        by[split]={}
        for cap in budgets:
            b=[r["trace"]["budgets"][str(cap)] for r in cohort]
            by[split][str(cap)]={
                "prompts":len(cohort),
                "default_eos_terminated":sum(item["default_eos_terminated"] for item in b),
                "hit_cap":sum(item["would_hit_cap"] for item in b),
                "im_end_seen":sum(item["im_end_seen_by_cap"] for item in b),
                "mean_tokens":mean(item["completion_tokens"] for item in b),
            }
    return {
        "experiment":"TRAIN-009B",
        "kind":"frozen_generation_only_terminal_audit",
        "model":MODEL,
        "model_revision":REVISION,
        "sft_adapter_sha256":SFT_SHA,
        "seed":SAMPLE_SEED,
        "decode":"Qwen3 SFT sampled temperature0.8 top_p0.95 top_k0; one 512-token rollout per source prompt",
        "default_eos_id":EXPECTED_MODEL_EOS,
        "im_end_probe_id":ADDITIONAL_IM_END,
        "source_prompts":{"train_smoke":16,"sft_heldout":16},
        "source_label_leakage":False,
        "no_training":True,
        "no_hidden_test_use":True,
        "metrics":by,
        "limits":[
            "Derived 96/256 caps from the exact same sampled 512-token prefix; not independent generated responses.",
            "Default Qwen3-1.7B Base EOS is endoftext(151643). im_end(151645) is separately counted but NOT silently treated as EOS.",
            "All measures are terminal rates on frozen source prompts only, not answer correctness or generalization.",
            "Prompt selection fixed by SHA42, not cherry-picked based on model EOS results.",
        ],
    }


def _hash_rows(pairs:list[tuple[str,dict[str,str]]])->str:
    return hashlib.sha256(("\n".join(
        f"{split}|{row['prompt_sha256']}" for split,row in pairs)+"\n").encode()).hexdigest()


def run(*,root:Path, data_root:Path,sft_dir:Path,
        max_generation:int=512)->dict[str,Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
    if max_generation!=512:
        raise ValueError("frozen TRAIN-009B always probes one 512-token rollout")
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("real single CUDA card required for SFT generation audit")
    train,train_audit=frozen_prompt_pool(root=data_root,seed=SAMPLE_SEED,max_train_prompts=16)
    validation=frozen_holdout(data_root,count=16)
    pairs=[("train_smoke",x) for x in train]+[("sft_heldout",x) for x in validation]
    assert len({r["prompt_sha256"] for _,r in pairs})==32, "cross-cohort prompt overlap"
    model_file=sft_dir/"adapter_model.safetensors"
    if sha256(model_file)!=SFT_SHA:
        raise ValueError("SFT adapter SHA failed")
    source={
        "experiment":"TRAIN-009B",
        "sft_sha":SFT_SHA,
        "source_train_sha":train_audit["input_sha256"]["sft_train.jsonl"],
        "source_valid_sha":SFT_VALID_SHA,
        "reward_train_sha":train_audit["input_sha256"]["dpo_train.jsonl"],
        "reward_valid_sha":train_audit["input_sha256"]["dpo_validation.jsonl"],
        "prompt_hash_pairs":_hash_rows(pairs),
        "selection":"seed42 top SHA16 from TRAIN009A smoke and SFT heldout; excl RM train/test",
        "decode":{"temperature":0.8,"top_p":0.95,"top_k":0,"max_new_tokens":512},
    }
    root.mkdir(parents=True,exist_ok=True)
    identity_path=root/"frozen_inputs.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text())!=source:
            raise ValueError("resume identity diverged")
    elif (root/"trace_metrics.jsonl").exists() or (root/"summary.json").exists():
        raise ValueError("existing unpinned results; do not override")
    else:identity_path.write_text(json.dumps(source,indent=2)+"\n")
    log=root/"trace_metrics.jsonl"
    cache={}
    if log.exists():
        for line in log.read_text().splitlines():
            if not line.strip():continue
            row=json.loads(line)
            key=(row["split"],row["prompt_sha256"])
            if key in cache:raise ValueError("duplicate audit trace")
            cache[key]=row
    approved={(split,row["prompt_sha256"]) for split,row in pairs}
    if not set(cache)<=approved:
        raise ValueError("stale trace entered frozen cohort")
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,trust_remote_code=True)
    if tokenizer.eos_token_id!=EXPECTED_MODEL_EOS:
        raise ValueError("Qwen3 default EOS changed")
    if tokenizer.convert_tokens_to_ids("<|im_end|>")!=ADDITIONAL_IM_END:
        raise ValueError("Qwen3 special IM_END token changed")
    base=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,
        dtype=torch.bfloat16,trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),
        device_map={"":torch.cuda.current_device()})
    model=PeftModel.from_pretrained(base,str(sft_dir),is_trainable=False)
    model.eval()
    for i,(split,row) in enumerate(pairs):
        key=(split,row["prompt_sha256"])
        if key in cache:continue
        ids=tokenizer(text=row["prompt"],return_tensors="pt")["input_ids"].to(model.device)
        if ids.shape[1]>480:
            raise ValueError("frozen sample prompt length exceeds SFT policy budget; refuse silent truncation")
        torch.manual_seed(SAMPLE_SEED+i)
        torch.cuda.manual_seed_all(SAMPLE_SEED+i)
        with torch.inference_mode():
            out=model.generate(
                input_ids=ids,do_sample=True,temperature=0.8,
                top_p=0.95,top_k=0,max_new_tokens=max_generation,
                eos_token_id=EXPECTED_MODEL_EOS,pad_token_id=EXPECTED_MODEL_EOS,
                use_cache=True,
            )
        tokens=out[0,ids.shape[1]:].tolist()
        trace=boundary_stats(tokens)
        rec={"split":split,"prompt_sha256":row["prompt_sha256"],
             "prompt_tokens":int(ids.shape[1]),"trace":trace}
        with log.open("a") as f:f.write(json.dumps(rec)+"\n")
        cache[key]=rec
        if (i+1)%4==0:
            print(f"TRAIN-009B real GPU EOS stats {i+1}/{len(pairs)}",flush=True)
    ordered=[cache[(split,row["prompt_sha256"])] for split,row in pairs]
    result=summarize(ordered)
    result["frozen_inputs"]=source
    result["actual_trace_sha256"]=sha256(log)
    result["gpu"]={"name":torch.cuda.get_device_name(0),"peak_reserved_bytes":torch.cuda.max_memory_reserved()}
    if (root/"summary.json").exists():
        if json.loads((root/"summary.json").read_text())!=result:
            raise ValueError("cannot replace changed summary")
    else:(root/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path("artifacts/train009b-termination-audit"))
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--sft-dir",type=Path,required=True)
    a=p.parse_args()
    run(root=a.root,data_root=a.data_root,sft_dir=a.sft_dir)


if __name__=="__main__":
    main()
