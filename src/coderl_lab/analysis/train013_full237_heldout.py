"""TRAIN-013 retrospective heldout preference diagnostic for full237 BF16/FP32.

Original TRAIN-012 BF16 full237 and new TRAIN-013 FP32 full237 plus initial SFT are evaluated
on identical 256 UltraFeedback preference pairs. The heldout was inspected
during TRAIN-012; this is exploratory, NOT a new blind quality claim.
No training or gradient update occurs in this script.
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

from coderl_lab.train.train012_preflight import digest

MODEL="Qwen/Qwen2.5-7B-Instruct"
REVISION="a09a35458c702b33eeacc393d103063234e8bc28"
SFT_SHA="25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8"
HELDOUT_SHA="9d8a62dee49e9841add47a2ed27e485926c41e988ac06185ccd23b0df4e2e450"
MAX_LENGTH=512
BETA=.1
ARMS=("sft","bf16","fp32")


def task_sha(x: Any) -> str:
    return hashlib.sha256(str(x).encode("utf-8")).hexdigest()


def bootstrap(effects: list[float], n: int=20000, seed:int=42) -> dict:
    if not effects:
        raise ValueError("no evaluable pairs")
    rng=random.Random(seed)
    count=len(effects)
    results=sorted(sum(effects[rng.randrange(count)] for _ in range(count))/count for _ in range(n))
    return {
        "observed_mean":sum(effects)/count,
        "ci95_low":results[int(.025*n)],
        "ci95_high":results[int(.975*n)-1],
        "iterations":n,"seed":seed,
    }


def summarize(rows:list[dict]) -> dict:
    evaluated=[x for x in rows if x.get("scorable") is True]
    result={
        "total_pairs":len(rows),"scorable_pairs":len(evaluated),
        "unscorable_context512":len(rows)-len(evaluated),
    }
    if not evaluated:
        raise ValueError("no scoreable preference pairs")
    margins={arm:[float(r[f"{arm}_chosen_logp"])-float(r[f"{arm}_rejected_logp"])
                  for r in evaluated] for arm in ARMS}
    for arm in ARMS:
        result[arm]={
            "raw_chosen_preferred":sum(m>0 for m in margins[arm]),
            "raw_preference_accuracy":sum(m>0 for m in margins[arm])/len(evaluated),
            "mean_chosen_minus_rejected_logp":sum(margins[arm])/len(evaluated),
        }
    for arm in ("bf16","fp32"):
        old=margins["sft"]
        new=margins[arm]
        d=[BETA*(x-y) for x,y in zip(new,old)]
        result[f"{arm}_minus_initial_sft"]={
            "mean_implicit_reward_margin":sum(d)/len(d),
            "positive_implicit_reward_margin_fraction":sum(z>0 for z in d)/len(d),
            "preference_wrong_to_correct":sum(y<=0 and x>0 for x,y in zip(new,old)),
            "preference_correct_to_wrong":sum(y>0 and x<=0 for x,y in zip(new,old)),
            "paired_implicit_margin_bootstrap":bootstrap(d),
        }
    bp=margins["bf16"]; fp=margins["fp32"]
    delta=[BETA*(f-b) for f,b in zip(fp,bp)]
    flips=[int(f>0)-int(b>0) for f,b in zip(fp,bp)]
    result["fp32_minus_bf16"]={
        "implicit_reward_margin_bootstrap":bootstrap(delta),
        "raw_chosen_preference_accuracy_delta_bootstrap":bootstrap(flips),
        "bf16_wrong_fp32_correct":sum(b<=0 and f>0 for b,f in zip(bp,fp)),
        "bf16_correct_fp32_wrong":sum(b>0 and f<=0 for b,f in zip(bp,fp)),
    }
    return result


def frozen_sources(preference_path:Path, sft:Path, bf16:Path, fp32:Path, report:Path, max_samples:int) -> dict:
    """Compare original TRAIN-012 full237 BF16 vs TRAIN-013 matched full237 FP32."""
    freeze=json.loads(report.read_text(encoding="utf-8"))
    if freeze.get("experiment")!="TRAIN-013-FP32-FULL237-ONLY-PRECISION" or freeze.get("status")!="frozen_pre_gpu":
        raise ValueError("not the preregistered full237 precision freeze")
    if digest(preference_path)!=HELDOUT_SHA or digest(sft/"adapter_model.safetensors")!=SFT_SHA:
        raise ValueError("source heldout data or SFT adapter changed")
    old=json.loads((bf16/"run_summary.json").read_text(encoding="utf-8"))
    new=json.loads((fp32/"run_summary.json").read_text(encoding="utf-8"))
    if digest(bf16/"run_summary.json")!=freeze["prior_full237_run_summary_sha256"]:
        raise ValueError("prior full237 BF16 run identity changed")
    for run,arm in ((old,"bf16"),(new,"fp32")):
        if (
            run.get("global_step")!=237
            or run.get("trainer_max_steps")!=237
            or run.get("final_epoch")!=1.0
            or run.get("num_pairs")!=2048
            or run.get("sft_adapter_sha256")!=SFT_SHA
            or run.get("quantization",{}).get("mode")!="nf4"
            or run.get("checkpoint_policy",{}).get("identity",{}).get("training_data_sha256")!=freeze["training_data_sha256"]
        ):
            raise ValueError("not a complete matched full237 BF16/FP32 DPO training run")
    if old.get("trainable_adapter_precision") is not None:
        raise ValueError("old BF16 baseline must be unpatched TRL-forced BF16")
    if (
        new.get("trainable_adapter_precision",{}).get("requested")!="fp32"
        or new["trainable_adapter_precision"].get("trainable_dtypes_after")!=["torch.float32"]
        or new["trainable_adapter_precision"].get("paired_ref_initially_identical") is not True
    ):
        raise ValueError("new full237 model did not use verified FP32 default adapter")
    if old["checkpoint_policy"]["identity"]["config_sha256"]!=freeze["old_training_config_sha256"]:
        raise ValueError("original DPO formal config SHA changed")
    if new["checkpoint_policy"]["identity"]["config_sha256"]!=freeze["new_training_config_sha256"]:
        raise ValueError("FP32 full237 config SHA changed")
    observed={}
    for arm,folder,summary in (("bf16",bf16,old),("fp32",fp32,new)):
        value=digest(folder/"adapter_model.safetensors")
        if value!=summary["output_adapter_sha256"]:
            raise ValueError(f"frozen {arm} output Adapter SHA differs from trainer record")
        observed[arm]=value
    return {
        "experiment":"TRAIN-013-FULL237-FP32-VS-BF16-HELDOUT",
        "verification":"old frozen TRAIN-012 full237 BF16 compared with identical TRAIN-013 FP32 full237",
        "dataset_sha256":HELDOUT_SHA,
        "sft_sha256":SFT_SHA,
        "bf16_sha256":observed["bf16"],
        "fp32_sha256":observed["fp32"],
        "precision_full237_freeze_sha256":digest(report),
        "source_code_sha256":digest(Path(__file__)),
        "truncation_length":MAX_LENGTH,
        "requested_pairs":max_samples,
        "calculation":"sum conditional logp of overlapping completion tokens in concatenated prompt+answer; beta .1",
        "reference_frozen":True,
        "research_status":"retrospective full237 precision control; previously inspected preference holdout, not blind",
    }

def make_tokens(tokenizer,prompt:str,completion:str,device) -> tuple[Any,Any,list[int]]:
    enc=tokenizer(
        prompt+completion,return_tensors="pt",return_offsets_mapping=True,
        truncation=True,max_length=MAX_LENGTH,add_special_tokens=True,
    )
    offsets=enc.pop("offset_mapping")[0].tolist()
    positions=[
        i for i,(left,right) in enumerate(offsets)
        if i>0 and right>len(prompt) and right>left
    ]
    return enc["input_ids"].to(device),enc["attention_mask"].to(device),positions


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--heldout",type=Path,required=True)
    p.add_argument("--sft",type=Path,required=True)
    p.add_argument("--bf16",type=Path,required=True)
    p.add_argument("--fp32",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--max-pairs",type=int,default=256)
    a=p.parse_args()
    if a.max_pairs<1 or a.max_pairs>256:
        raise ValueError("max-pairs out of frozen256")
    identity=frozen_sources(a.heldout,a.sft,a.bf16,a.fp32,a.report,a.max_pairs)
    rows=[json.loads(line) for line in a.heldout.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows)!=256 or len({r["task_id"] for r in rows})!=256:
        raise ValueError("not the 256 frozen UltraFeedback heldout pairs")
    a.output_dir.mkdir(parents=True,exist_ok=True)
    marker=a.output_dir/"identity.json"
    if marker.exists():
        if json.loads(marker.read_text())!=identity:
            raise ValueError("heldout benchmark checkpoint identity changed")
    else:
        marker.write_text(json.dumps(identity,indent=2)+"\n")
    checkpoint=a.output_dir/"checkpoint.jsonl"
    done={}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if not line.strip():continue
            row=json.loads(line);idx=int(row["index"])
            if idx in done or idx not in range(a.max_pairs):
                raise ValueError("duplicate/unrecognized heldout pair")
            if row["task_id_sha256"]!=task_sha(rows[idx]["task_id"]):
                raise ValueError("heldout checkpoint task id drift")
            done[idx]=row
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig

    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True,use_fast=True)
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    backbone=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,
                                                  dtype=torch.bfloat16,device_map={"":0},
                                                  quantization_config=q)
    model=PeftModel.from_pretrained(backbone,str(a.sft),adapter_name="sft",is_trainable=False)
    model.load_adapter(str(a.bf16),adapter_name="bf16",is_trainable=False)
    model.load_adapter(str(a.fp32),adapter_name="fp32",is_trainable=False)
    model.eval();model.config.use_cache=False

    def total_log_prob(inp,mask,positions)->float:
        with torch.inference_mode():
            logits=model(input_ids=inp,attention_mask=mask,use_cache=False).logits
            x=logits[:,[i-1 for i in positions],:].float()
            labels=inp[:,positions]
            v=F.cross_entropy(x.transpose(1,2),labels,reduction="sum")
            result=-float(v.item())
            del logits,x,labels,v
        return result
    started=time.monotonic()
    for idx in range(a.max_pairs):
        if idx in done:continue
        row=rows[idx]
        inputs={cand:make_tokens(tokenizer,row["prompt"],row[cand],model.device)
                for cand in ("chosen","rejected")}
        n_ch=len(inputs["chosen"][2])
        n_rj=len(inputs["rejected"][2])
        record={
            "index":idx,"task_id_sha256":task_sha(row["task_id"]),
            "scorable":bool(n_ch and n_rj),
            "chosen_target_tokens":n_ch,"rejected_target_tokens":n_rj,
        }
        if record["scorable"]:
            for arm in ARMS:
                model.set_adapter(arm)
                for cand in ("chosen","rejected"):
                    ids,mask,positions=inputs[cand]
                    record[f"{arm}_{cand}_logp"]=total_log_prob(ids,mask,positions)
        else:
            record["reason"]="one_or_both_completions_outside_512_token_window"
        with checkpoint.open("a",encoding="utf-8") as h:
            h.write(json.dumps(record)+"\n")
            h.flush()
        done[idx]=record
        if (idx+1)%16==0 or idx+1==a.max_pairs:
            print(f"TRAIN-013 heldout paired arms: {idx+1}/{a.max_pairs}",flush=True)
    if set(done)!=set(range(a.max_pairs)):
        raise ValueError("incomplete heldout checkpoint")
    summary={
        **identity,"status":"completed",
        "full256":a.max_pairs==256,
        "metrics":summarize([done[i] for i in range(a.max_pairs)]),
        "checkpoint_sha256":digest(checkpoint),
        "gpu_peak_allocated_mib":round(torch.cuda.max_memory_allocated()/2**20,2),
        "execution_seconds":round(time.monotonic()-started,2),
        "limitations":"Previously examined public historical preference split, not novel blind test. The score is total completion-token likelihood, length-sensitive, not independent human ratings.",
    }
    result_path=a.output_dir/"summary.json"
    if result_path.exists():
        if json.loads(result_path.read_text())!=summary:
            raise FileExistsError("refusing scoring report overwrite")
    else:
        result_path.write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
