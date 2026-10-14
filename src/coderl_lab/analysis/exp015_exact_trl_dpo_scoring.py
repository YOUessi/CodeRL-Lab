"""EXP-015: recompute SFT/BF16 DPO/FP32 DPO logP using ACTUAL TRL DPO tokens.

This is a retrospective implementation audit, not a new or blind human
preference benchmark. TRL chosen/rejected EOS, prefix split and keep_start
collator completion masks are re-used verbatim. NF4 BF16 three adapters share
the exact frozen 7B backbone and each sample is scored sequentially on GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from coderl_lab.analysis.exp015_dpo_token_contract import (
    DATA_SHA, MODEL,REVISION,MAX_LENGTH,digest,read_frozen,
)
SFT_SHA="25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8"
BF16_SHA="d586d9ef32377189aa7d79b70ea5e37b5d5f26b930a4f60266ca024213f1a185"
FP32_SHA="471703e1cec7fd322219133c10e09fa62aab2f6950b896688c7c073de1543281"
ARMS=("sft","bf16","fp32")
BETA=.1

def task_sha(q)->str:
    return hashlib.sha256(str(q).encode()).hexdigest()

def dataset_batch(tokenizer,record):
    from trl.trainer.dpo_trainer import _tokenize,DataCollatorForPreference
    prompt=record["prompt"]
    eos=tokenizer.eos_token
    if not eos:raise ValueError("eos unavailable")
    def with_eos(s): return s if s.endswith(eos) else s+eos
    prefix=_tokenize(tokenizer,prompt)["input_ids"]
    chosen=_tokenize(tokenizer,prompt+with_eos(record["chosen"]))["input_ids"]
    rejected=_tokenize(tokenizer,prompt+with_eos(record["rejected"]))["input_ids"]
    train_row={
        "prompt_ids":prefix,
        "chosen_ids":chosen[len(prefix):],
        "rejected_ids":rejected[len(prefix):],
    }
    collator=DataCollatorForPreference(
        pad_token_id=tokenizer.pad_token_id,
        max_length=MAX_LENGTH,truncation_mode="keep_start")
    return collator([train_row]),len(prefix)

def confidence(values:list[float],rep:int=20000,seed:int=42)->dict:
    if not values:raise ValueError("no values for paired bootstrap")
    n=len(values);rng=random.Random(seed)
    v=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(rep))
    return {"observed":sum(values)/n,"ci95_low":v[int(.025*rep)],
            "ci95_high":v[int(.975*rep)-1],"iterations":rep,"seed":seed}

def summarize(checkpoint:list[dict])->dict:
    valid=[r for r in checkpoint if r.get("scoreable")]
    if not valid:raise ValueError("no scoreable samples")
    data={"total":len(checkpoint),"scoreable":len(valid),"unscorable":len(checkpoint)-len(valid)}
    acc={}
    margin={}
    old_acc={}
    for arm in ARMS:
        margin[arm]=[x[f"{arm}_chosen_logp"]-x[f"{arm}_rejected_logp"] for x in valid]
        oldmargin=[x[f"old_{arm}_chosen_logp"]-x[f"old_{arm}_rejected_logp"] for x in valid]
        acc[arm]=sum(m>0 for m in margin[arm])
        old_acc[arm]=sum(m>0 for m in oldmargin)
        data[f"{arm}_train_exact_chosen_preferred"]=acc[arm]
        data[f"{arm}_old_offset_chosen_preferred"]=old_acc[arm]
        data[f"{arm}_old_vs_exact_judgment_flip"]=sum((x>0)!=(y>0) for x,y in zip(oldmargin,margin[arm]))
        data[f"{arm}_total_logp_margin_delta_exact_minus_old"]=confidence(
            [x-y for x,y in zip(margin[arm],oldmargin)]
        )
    for arm in ("bf16","fp32"):
        data[arm+"_minus_sft_implicit_reward_margin"]=confidence(
            [BETA*(a-b) for a,b in zip(margin[arm],margin["sft"])]
        )
        data[arm+"_minus_sft_raw_accuracy"]=confidence(
            [int(a>0)-int(b>0) for a,b in zip(margin[arm],margin["sft"])]
        )
    data["fp32_minus_bf16_implicit_reward_margin"]=confidence([
        BETA*(a-b) for a,b in zip(margin["fp32"],margin["bf16"])
    ])
    data["fp32_minus_bf16_raw_accuracy"]=confidence([
        int(a>0)-int(b>0) for a,b in zip(margin["fp32"],margin["bf16"])
    ])
    data["qualification"]="Retrospective exact TRL scoring; dataset previously examined, no new heldout generalization"
    return data

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--heldout",required=True,type=Path)
    p.add_argument("--cpu-audit",required=True,type=Path)
    p.add_argument("--cpu-details",required=True,type=Path)
    p.add_argument("--old-checkpoint",required=True,type=Path)
    p.add_argument("--sft",required=True,type=Path)
    p.add_argument("--bf16",required=True,type=Path)
    p.add_argument("--fp32",required=True,type=Path)
    p.add_argument("--output-dir",required=True,type=Path)
    p.add_argument("--max-pairs",type=int,default=256)
    a=p.parse_args()
    if not 1<=a.max_pairs<=256:raise ValueError("max-pairs out of range")
    frozen=read_frozen(a.heldout)
    cpu=json.loads(a.cpu_audit.read_text())
    if (cpu.get("status")!="completed_cpu_full_256"
        or digest(a.cpu_details)!=cpu["details_sha256"]):
        raise ValueError("EXP015 CPU token audit changed")
    adapters={"sft":a.sft,"bf16":a.bf16,"fp32":a.fp32}
    expected={"sft":SFT_SHA,"bf16":BF16_SHA,"fp32":FP32_SHA}
    for arm,path in adapters.items():
        if digest(path/"adapter_model.safetensors")!=expected[arm]:
            raise ValueError(f"model Adapter SHA changed: {arm}")
    old=[json.loads(l) for l in a.old_checkpoint.read_text().splitlines() if l.strip()]
    if len(old)!=256 or {x["index"] for x in old}!=set(range(256)):
        raise ValueError("previous diagnostic not frozen full 256")
    identity={
        "experiment":"EXP-015-EXACT-TRL-DPO-GPU-SCORING",
        "heldout_sha256":DATA_SHA,
        "source_cpu_audit_sha256":digest(a.cpu_audit),
        "old_scoring_checkpoint_sha256":digest(a.old_checkpoint),
        "adapters_sha256":expected,
        "model":MODEL,"revision":REVISION,
        "max_length":MAX_LENGTH,"append_eos":True,"truncation":"keep_start",
        "dataset_preprocessor":"actual TRL1.14.1 DPOTrainer _tokenize/DataCollatorForPreference",
        "requested_pairs":a.max_pairs,
        "code_sha256":digest(Path(__file__)),
        "never_retrained_models":True,
    }
    a.output_dir.mkdir(parents=True,exist_ok=True)
    marker=a.output_dir/"identity.json"
    if marker.exists():
        if json.loads(marker.read_text())!=identity:raise ValueError("scoring identity drift")
    else:marker.write_text(json.dumps(identity,indent=2)+"\n")
    checkpoint=a.output_dir/"checkpoint.jsonl"
    done={}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if not line.strip():continue
            row=json.loads(line);i=int(row["index"])
            if i in done or i not in range(a.max_pairs) or row["task_id_sha256"]!=task_sha(frozen[i]["task_id"]):
                raise ValueError("old checkpoint mismatched or duplicated")
            done[i]=row
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    import trl,transformers
    if trl.__version__!="1.14.1" or transformers.__version__!="5.18.0":
        raise ValueError("TRL/Transformers changed")
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True,use_fast=True)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    base=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=REVISION,local_files_only=True,device_map={"":0},dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16))
    model=PeftModel.from_pretrained(base,str(a.sft),adapter_name="sft",is_trainable=False)
    model.load_adapter(str(a.bf16),adapter_name="bf16",is_trainable=False)
    model.load_adapter(str(a.fp32),adapter_name="fp32",is_trainable=False)
    model.eval();model.config.use_cache=False
    def logp(ids,mask,mask_completion):
        selected=[
            i for i in range(1,len(mask_completion))
            if mask_completion[i].item()==1 and mask[0,i].item()==1
        ]
        if not selected:return None
        with torch.inference_mode():
            output=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
            picked=output[:,[i-1 for i in selected],:].float()
            target=ids[:,selected]
            loss=F.cross_entropy(picked.transpose(1,2),target,reduction="sum")
            answer=-float(loss.item())
            del output,picked,loss
        return answer
    started=time.monotonic()
    for i,row in enumerate(frozen[:a.max_pairs]):
        if i in done:continue
        b,prompt_len=dataset_batch(tokenizer,row)
        old_row=old[i]
        if old_row["task_id_sha256"]!=task_sha(row["task_id"]):
            raise ValueError("old old-scoring task ordering differs")
        detail={"index":i,"task_id_sha256":task_sha(row["task_id"]),
                "trl_prompt_len":prompt_len,"scoreable":False}
        valid=prompt_len<512 and all(
            int(b["completion_mask"][arm_index].sum().item())>0
            for arm_index in range(2))
        if valid:
            detail["scoreable"]=True
            for arm in ARMS:
                model.set_adapter(arm)
                for which,j in (("chosen",0),("rejected",1)):
                    ids=b["input_ids"][j:j+1].to(model.device)
                    mask=b["attention_mask"][j:j+1].to(model.device)
                    comp=b["completion_mask"][j].to(model.device)
                    detail[f"{arm}_{which}_logp"]=logp(ids,mask,comp)
                    detail[f"old_{arm}_{which}_logp"]=old_row[f"{arm}_{which}_logp"]
        else:
            detail["reason"]="no_completion_after_exact_trl_keep_start_512"
        with checkpoint.open("a",encoding="utf-8") as fh:
            fh.write(json.dumps(detail,ensure_ascii=False)+"\n");fh.flush()
        done[i]=detail
        if (i+1)%16==0 or i+1==a.max_pairs:
            print(f"EXP015 exact training-collator logP {i+1}/{a.max_pairs}",flush=True)
    if set(done)!=set(range(a.max_pairs)):
        raise RuntimeError("EXP015 GPU scoring checkpoint incomplete")
    result={
        **identity,
        "status":"completed",
        "full256":a.max_pairs==256,
        "metrics":summarize([done[i] for i in range(a.max_pairs)]),
        "checkpoint_sha256":digest(checkpoint),
        "gpu_peak_allocated_mib":round(torch.cuda.max_memory_allocated()/2**20,2),
        "elapsed_seconds":round(time.monotonic()-started,2),
        "limitations":"This is the historical heldout rescored to match TRL training, not prospective human preference validation.",
    }
    (a.output_dir/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":main()
