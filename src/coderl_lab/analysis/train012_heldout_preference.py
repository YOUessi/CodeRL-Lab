"""TRAIN-012 7B independently held-out preference likelihood audit.

Uses the frozen 256 UltraFeedback test-preference pairs. Source SFT and
new DPO policy use identical Qwen2.5-7B backbone, NF4, BF16 and tokenizer.
The comparison performs single-sequence, completion-token-only log-prob
scoring to avoid Trainer evaluation's large FP32 logits OOM. This is
a descriptive preference likelihood diagnostic, not a human rating.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

MODEL = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
HELDOUT_SHA = "9d8a62dee49e9841add47a2ed27e485926c41e988ac06185ccd23b0df4e2e450"
SFT_SHA = "25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8"
DPO_SHA = "d586d9ef32377189aa7d79b70ea5e37b5d5f26b930a4f60266ca024213f1a185"
BETA = 0.1
MAX_LENGTH = 512

def file_sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):
            h.update(chunk)
    return h.hexdigest()

def validate_rows(rows: list[dict]) -> list[dict]:
    if len(rows)!=256:
        raise ValueError("heldout must contain 256 fixed pairs")
    ids=set()
    for row in rows:
        if not all(k in row for k in ("task_id","prompt","chosen","rejected")):
            raise ValueError("missing heldout field")
        if any(not isinstance(row[k],str) or not row[k].strip()
               for k in ("prompt","chosen","rejected")):
            raise ValueError("empty heldout preference field")
        if row["chosen"]==row["rejected"]:
            raise ValueError("identical chosen/rejected pair")
        if row["task_id"] in ids:
            raise ValueError("duplicate heldout task ID")
        ids.add(row["task_id"])
    return rows

def comparison(values: list[dict]) -> dict:
    scored=[r for r in values if r.get("scorable") is True]
    all_count=len(values)
    if not scored:
        raise ValueError("no scorable chosen/rejected pairs")
    base=[r["sft_chosen_logp"]-r["sft_rejected_logp"] for r in scored]
    newer=[r["dpo_chosen_logp"]-r["dpo_rejected_logp"] for r in scored]
    margin=[
        BETA*(d-b) for d,b in zip(newer,base)
    ]
    # Compares the log-ratio for the very same chosen/rejected strings;
    # positive DPO delta is the direction optimized by DPO.
    raw={
        "heldout_pairs_total":all_count,
        "scorable_pairs":len(scored),
        "unscorable_512_context_pairs":all_count-len(scored),
        "scorable_fraction":len(scored)/all_count,
        "sft_raw_chosen_preference_accuracy":sum(v>0 for v in base)/len(scored),
        "dpo_raw_chosen_preference_accuracy":sum(v>0 for v in newer)/len(scored),
        "sft_mean_raw_logp_chosen_minus_rejected":sum(base)/len(scored),
        "dpo_mean_raw_logp_chosen_minus_rejected":sum(newer)/len(scored),
        "dpo_relative_to_sft_implicit_reward_margin_mean":sum(margin)/len(scored),
        "dpo_implicit_reward_chosen_preferred_fraction":sum(v>0 for v in margin)/len(scored),
        "sft_to_dpo_wrong_to_correct":sum(b<=0 and d>0 for b,d in zip(base,newer)),
        "sft_to_dpo_correct_to_wrong":sum(b>0 and d<=0 for b,d in zip(base,newer)),
    }
    rng=random.Random(42)
    def paired_ci(deltas:list[float], nboot:int=20000) -> dict:
        n=len(deltas)
        samples=sorted(
            sum(deltas[rng.randrange(n)] for _ in range(n))/n
            for _ in range(nboot)
        )
        return {
            "observed":sum(deltas)/n,
            "ci95_low":samples[int(.025*nboot)],
            "ci95_high":samples[int(.975*nboot)-1],
            "seed":42,"nboot":nboot,"unit":"pair",
        }
    raw["paired_bootstrap"]={
        "raw_preference_accuracy_delta":paired_ci(
            [int(d>0)-int(b>0) for b,d in zip(base,newer)]
        ),
        "implicit_reward_margin":paired_ci(margin),
    }
    return raw

def completion_batch(tok, prompt:str, completion:str, device) -> tuple[Any,Any,list[int]]:
    source=prompt+completion
    encoded=tok(
        source,add_special_tokens=True,truncation=True,
        max_length=MAX_LENGTH,return_tensors="pt",return_offsets_mapping=True,
    )
    offsets=encoded.pop("offset_mapping")[0].tolist()
    positions=[
        i for i,(begin,end) in enumerate(offsets)
        if i>0 and end>len(prompt) and end>begin
    ]
    return encoded["input_ids"].to(device),encoded["attention_mask"].to(device),positions

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--heldout",required=True,type=Path)
    p.add_argument("--sft-adapter",required=True,type=Path)
    p.add_argument("--dpo-adapter",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--max-examples",type=int,default=256)
    args=p.parse_args()
    if args.max_examples not in range(1,257):
        raise ValueError("max-examples outside range 1..256")
    if file_sha(args.heldout)!=HELDOUT_SHA:
        raise ValueError("heldout SHA mismatch")
    for adapter,expected in ((args.sft_adapter,SFT_SHA),(args.dpo_adapter,DPO_SHA)):
        if file_sha(adapter/"adapter_model.safetensors")!=expected:
            raise ValueError("adapter SHA mismatch")
    rows=validate_rows([
        json.loads(line) for line in args.heldout.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ])
    identity={
        "experiment":"TRAIN-012-HELDOUT-PAIRWISE-LOGP",
        "source_model":MODEL,"revision":REVISION,
        "heldout_sha256":HELDOUT_SHA,"sft_adapter_sha256":SFT_SHA,
        "dpo_adapter_sha256":DPO_SHA,
        "max_tokens":MAX_LENGTH,
        "max_examples":args.max_examples,
        "score":"sum log prob of completion-overlapping tokens in concatenated prompt/completion",
        "compute":"NF4 double quant BF16 both adapters; same backbone",
        "no_target":"prompt length >=512 means no completion token: report as unscorable",
        "beta":BETA,
        "audit_source_sha256":file_sha(Path(__file__)),
        "uses_train_data":False,
    }
    args.out.mkdir(parents=True,exist_ok=True)
    identity_path=args.out/"identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8"))!=identity:
            raise ValueError("frozen identity changed")
    else:
        identity_path.write_text(json.dumps(identity,indent=2)+"\n",encoding="utf-8")
    checkpoint=args.out/"checkpoint.jsonl"
    done={}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if not line.strip():continue
            row=json.loads(line)
            idx=row["index"]
            if idx in done or idx not in range(args.max_examples):
                raise ValueError("duplicate/extraneous checkpoint")
            if row["task_id_sha256"]!=hashlib.sha256(str(rows[idx]["task_id"]).encode()).hexdigest():
                raise ValueError("task ID checkpoint SHA mismatch")
            done[idx]=row

    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
    from peft import PeftModel
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True,use_fast=True)
    quant=BitsAndBytesConfig(
        load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=REVISION,local_files_only=True,device_map={"":0},
        quantization_config=quant,dtype=torch.bfloat16,
    )
    model=PeftModel.from_pretrained(base,str(args.sft_adapter),adapter_name="sft",is_trainable=False)
    model.load_adapter(str(args.dpo_adapter),adapter_name="dpo",is_trainable=False)
    model.eval()
    model.config.use_cache=False
    def score(ids,mask,positions) -> float:
        selected=[i-1 for i in positions]
        with torch.inference_mode():
            logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
            chosen=logits[:,selected,:].float()
            target=ids[:,positions]
            nll=F.cross_entropy(
                chosen.transpose(1,2),target,reduction="sum"
            )
            value=-float(nll.item())
            del logits,chosen,nll
        return value
    started=time.monotonic()
    for idx in range(args.max_examples):
        if idx in done:continue
        row=rows[idx]
        fixed={"index":idx,"task_id_sha256":hashlib.sha256(
            str(row["task_id"]).encode()
        ).hexdigest()}
        tokenized={
            k:completion_batch(tok,row["prompt"],row[k],model.device)
            for k in ("chosen","rejected")
        }
        positions=[len(tokenized[k][2]) for k in ("chosen","rejected")]
        if not all(positions):
            entry={**fixed,"scorable":False,"reason":"no_completion_token_in_frozen_512_context",
                   "chosen_target_tokens":positions[0],"rejected_target_tokens":positions[1]}
        else:
            values={}
            for name in ("sft","dpo"):
                model.set_adapter(name)
                for candidate in ("chosen","rejected"):
                    ids,mask,selected=tokenized[candidate]
                    values[f"{name}_{candidate}_logp"]=score(ids,mask,selected)
            entry={**fixed,"scorable":True,
                   "chosen_target_tokens":positions[0],"rejected_target_tokens":positions[1],
                   **values}
        with checkpoint.open("a",encoding="utf-8") as handle:
            handle.write(json.dumps(entry,ensure_ascii=False)+"\n")
            handle.flush()
        done[idx]=entry
        if (idx+1)%16==0 or idx+1==args.max_examples:
            print(f"TRAIN-012 independent heldout: {idx+1}/{args.max_examples}",flush=True)
    if set(done)!=set(range(args.max_examples)):
        raise ValueError("evaluation incomplete")
    output={
        **identity,"status":"completed",
        "full_256":args.max_examples==256,
        "metrics":comparison([done[i] for i in range(args.max_examples)]),
        "scoring_checkpoint_sha256":file_sha(checkpoint),
        "gpu_peak_allocated_mib":round(torch.cuda.max_memory_allocated()/2**20,2),
        "elapsed_seconds_current_invocation":round(time.monotonic()-started,2),
        "limitations":"This scores likelihood of UltraFeedback chosen vs rejected completions on a frozen 512-token window. It is not judged response quality, and 512-truncated prompts cannot be scored.",
    }
    result=args.out/"summary.json"
    if result.exists():
        if json.loads(result.read_text())!=output:
            raise FileExistsError("previous summary differs")
    else:
        result.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(output,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
