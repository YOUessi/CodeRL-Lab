"""TRAIN-007D: frozen, paired SFT-vs-DPO preference likelihood evaluation.

Compare the two trained policies on EXACTLY the same UltraFeedback heldout
pairs. This computes policy log-likelihood ranking, not DPO reference-relative
reward accuracy, correctness, or hidden benchmark evaluation.

No data/model selection from outcomes. No model training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import yaml

MODEL="Qwen/Qwen3-1.7B-Base"
MODEL_REVISION="ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
ADAPTERS={
    "sft":"d8846aa5fb5fd6958d30a24611efd9fb99eb91469a60192937646b58557ce12d",
    "dpo":"3cb6ff9bd51e273b8c77bcf07f574a708b1ad2b089e53f9b61f4dc265c93ced4",
}
TRAIN_SHA="945a336244462d43a7c3e70774158a4c40e873b5ee4f6c93d73e0d1e82222b77"
HELDOUT_SHA="9d8a62dee49e9841add47a2ed27e485926c41e988ac06185ccd23b0df4e2e450"
MAX_LENGTH=1024
ARMS=("sft","dpo")
SEED=42
BOOTSTRAP_ITERATIONS=20000


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):
            h.update(b)
    return h.hexdigest()


def read_json(path:Path)->dict[str,Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path:Path)->list[dict[str,Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def verify_frozen_sources(*,manifest_path:Path,heldout_path:Path,
                          train_path:Path, sft_adapter:Path,dpo_adapter:Path,
                          sft_summary:Path,dpo_summary:Path)->dict[str,Any]:
    manifest=read_json(manifest_path)
    sft=read_json(sft_summary)
    dpo=read_json(dpo_summary)
    if any((
        manifest.get("track")!="A-posttraining",
        manifest.get("stage")!="dpo",
        manifest.get("full_scan") is not True,
        manifest.get("prompt_overlap")!=0,
        manifest["train"]["rows"]!=2048,
        manifest["validation"]["rows"]!=256,
        manifest["train"]["sha256"]!=TRAIN_SHA,
        manifest["validation"]["sha256"]!=HELDOUT_SHA,
        sha(train_path)!=TRAIN_SHA,
        sha(heldout_path)!=HELDOUT_SHA,
        sft.get("model")!=MODEL,
        dpo.get("model")!=MODEL,
        sft.get("requested_model_revision")!=MODEL_REVISION,
        dpo.get("requested_model_revision")!=MODEL_REVISION,
        sft.get("num_examples")!=4096,
        dpo.get("num_pairs")!=2048,
        dpo.get("num_heldout_preference_pairs")!=256,
        dpo.get("global_step")!=251,
        dpo.get("quantization",{}).get("mode")!="nf4",
        sft.get("quantization",{}).get("mode")!="nf4",
        sft.get("saved_adapter_sha256")!=ADAPTERS["sft"],
        dpo.get("sft_adapter_sha256")!=ADAPTERS["sft"],
        dpo.get("output_adapter_sha256")!=ADAPTERS["dpo"],
        sha(sft_adapter)!=ADAPTERS["sft"],
        sha(dpo_adapter)!=ADAPTERS["dpo"],
    )):
        raise ValueError("one or more frozen DPO/SFT/data/model SHA contracts changed")
    return {
        "experiment":"TRAIN-007D",
        "sft_adapter_sha256":ADAPTERS["sft"],
        "dpo_adapter_sha256":ADAPTERS["dpo"],
        "train_sha256":TRAIN_SHA,
        "heldout_sha256":HELDOUT_SHA,
        "manifest_sha256":sha(manifest_path),
        "model":MODEL,
        "model_revision":MODEL_REVISION,
        "max_length":MAX_LENGTH,
        "tokenization":"TRL DPO keep_start with EOS and prompt-masked completion labels",
        "raw_heldout_pairs":256,
        "no_private_or_hidden_tests":True,
        "no_optimizer_training":True,
    }


def encode_pair(tokenizer, raw:dict[str,str], *, max_length:int=MAX_LENGTH)->dict[str,Any] | None:
    prompt=str(raw["prompt"])
    if not (prompt.strip() and str(raw["chosen"]).strip() and str(raw["rejected"]).strip()):
        raise ValueError("malformed frozen heldout preference pair")
    prompt_ids=list(tokenizer(text=prompt)["input_ids"])
    if len(prompt_ids)>=max_length:
        return None
    outputs={"prompt_token_count":len(prompt_ids)}
    for label in ("chosen","rejected"):
        text=str(raw[label])
        if not text.endswith(tokenizer.eos_token):
            text+=tokenizer.eos_token
        # Exactly the TRL DPOTrainer logic for non-conversational rows:
        # tokenize prompt and prompt+completion then slice off len(prompt).
        combo=list(tokenizer(text=prompt+text)["input_ids"])
        suffix=combo[len(prompt_ids):]
        if not suffix:
            raise ValueError("prompt-boundary split yielded empty completion")
        tokens=(prompt_ids+suffix)[:max_length]
        mask=([False]*len(prompt_ids)+[True]*len(suffix))[:max_length]
        if not any(mask) or len(tokens)!=len(mask):
            raise ValueError("completion lost under keep_start truncation")
        outputs[label+"_ids"]=tokens
        outputs[label+"_mask"]=mask
        outputs[label+"_length"]=sum(mask)
        outputs[label+"_was_truncated"]=len(prompt_ids)+len(suffix)>max_length
    return outputs


def preference_accuracy(rows:list[dict[str,Any]],label:str)->dict[str,Any]:
    correct=sum(row[label+"_mean_delta"]>0 for row in rows)
    tie=sum(row[label+"_mean_delta"]==0 for row in rows)
    raw_correct=sum(row[label+"_total_delta"]>0 for row in rows)
    return {"correct":correct,"ties":tie,"pairs":len(rows),
            "mean_logp_preference_accuracy":correct/len(rows),
            "total_logp_preference_accuracy":raw_correct/len(rows)}


def paired_bootstrap(deltas:list[int],*,seed:int=SEED,iterations:int=BOOTSTRAP_ITERATIONS)->dict[str,Any]:
    if not deltas or set(deltas)-{-1,0,1}:
        raise ValueError("paired binary preference differences invalid")
    n=len(deltas)
    if iterations<1:
        raise ValueError("iterations must be > 0")
    rng=random.Random(seed)
    results=sorted(sum(deltas[rng.randrange(n)] for _ in range(n))/n for _ in range(iterations))
    return {"observed_difference":sum(deltas)/n,
            "ci95_low":results[int(0.025*(iterations-1))],
            "ci95_high":results[int(0.975*(iterations-1))],
            "iterations":iterations,"seed":seed}


def summarize(rows:list[dict[str,Any]])->dict[str,Any]:
    if len(rows)!=252:
        raise ValueError("must evaluate all 252 TRL-filtered heldout pairs before reporting")
    expected={"sft_mean_delta","dpo_mean_delta","sft_total_delta","dpo_total_delta"}
    if any(not expected.issubset(r) or
           not all(math.isfinite(r[k]) for k in expected) for r in rows):
        raise ValueError("missing or nonfinite policy log-probs")
    if len({str(x["row_id"]) for x in rows})!=252:
        raise ValueError("duplicate frozen heldout row")
    sft=preference_accuracy(rows,"sft")
    dpo=preference_accuracy(rows,"dpo")
    delta=[int(r["dpo_mean_delta"]>0)-int(r["sft_mean_delta"]>0) for r in rows]
    return {
        "experiment":"TRAIN-007D",
        "phase":"posthoc_matched_policy_loglikelihood_evaluation",
        "raw_validation_pairs":256,
        "effective_validation_pairs":252,
        "sft":sft,"dpo":dpo,
        "paired_mean_token_preference_accuracy_difference":
            paired_bootstrap(delta),
        "paired_accuracy_flips":{
            "sft_wrong_to_dpo_correct":sum(x==1 for x in delta),
            "sft_correct_to_dpo_wrong":sum(x==-1 for x in delta),
        },
        "limitations":[
            "Average completion token log-likelihood is a descriptive preference ranking diagnostic, not the DPO reference-relative reward metric.",
            "Responses vary in length; both mean-token and total-loglikelihood ranks are reported.",
            "Same heldout 252 pairs after the frozen TRL keep_start max_length1024 filter; no new train/test selection.",
            "This is one seeded matched retrospective validation, not human preference improvement, code correctness or PPO/RLHF."
        ],
    }


def policy_completion_logps(model,ids:list[int],mask:list[bool],*,device,chunk_tokens:int=96)->tuple[float,float,int]:
    import torch
    if len(ids)<2 or len(ids)!=len(mask) or not any(mask[1:]):
        raise ValueError("missing shifted completion target tokens")
    values=torch.tensor([ids],device=device,dtype=torch.long)
    with torch.inference_mode():
        logits=model(input_ids=values,use_cache=False).logits[0,:-1,:]
        positions=torch.tensor(mask[1:],device=device,dtype=torch.bool)
        target=values[0,1:][positions]
        selected=logits[positions]
        # Selected logits convert to float32 in bounded chunks to prevent
        # allocating a full [sequence x vocab] float32 log-softmax buffer.
        count=0
        total=0.0
        for start in range(0,target.numel(),chunk_tokens):
            target_part=target[start:start+chunk_tokens]
            fp32=selected[start:start+chunk_tokens].float()
            part=fp32.gather(1,target_part[:,None]).flatten()-torch.logsumexp(fp32,dim=1)
            total+=float(part.sum().item())
            count+=len(part)
        del logits,selected
    return total,total/count,count


def run(*,root:Path,sft_adapter:Path,dpo_adapter:Path,
        sft_summary:Path,dpo_summary:Path,heldout_path:Path,train_path:Path,
        manifest_path:Path)->dict[str,Any]:
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError("requires real CUDA inference; no mock values accepted")
    provenance=verify_frozen_sources(manifest_path=manifest_path,
        heldout_path=heldout_path,train_path=train_path,
        sft_adapter=sft_adapter,dpo_adapter=dpo_adapter,
        sft_summary=sft_summary,dpo_summary=dpo_summary)
    root.mkdir(parents=True,exist_ok=True)
    inputs_path=root/"frozen_inputs.json"
    if inputs_path.is_file():
        if read_json(inputs_path)!=provenance:
            raise ValueError("matched eval frozen inputs changed after start")
    else:
        if (root/"summary.json").exists() or any(root.glob("*.jsonl")):
            raise FileExistsError("existing matched evaluation without frozen inputs")
        inputs_path.write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_REVISION,trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    rows=read_jsonl(heldout_path)
    if len(rows)!=256:
        raise ValueError("full frozen heldout must have exactly 256 rows")
    inputs=[]
    for i,r in enumerate(rows):
        pair=encode_pair(tokenizer,r)
        if pair is not None:
            inputs.append((i,pair))
    if len(inputs)!=252:
        raise ValueError("actual TRL keep_start filtering count drifted")
    base=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=MODEL_REVISION,dtype=torch.bfloat16,trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),
        device_map={"":torch.cuda.current_device()},
    )
    base.config.use_cache=False
    model=PeftModel.from_pretrained(base,str(sft_adapter),adapter_name="sft",is_trainable=False)
    model.load_adapter(str(dpo_adapter),adapter_name="dpo",is_trainable=False)
    model.eval()
    process_file=root/"task_logps.jsonl"
    seen:dict[tuple[str,int],dict[str,Any]]={}
    if process_file.exists():
        for line in process_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():continue
            r=json.loads(line)
            key=(r["arm"],int(r["row_id"]))
            if key in seen:raise ValueError("duplicate cache rows")
            if r["arm"] not in ARMS or int(r["row_id"]) not in [i for i,_ in inputs]:
                raise ValueError("cached evaluation mismatches source")
            seen[key]=r
    for arm in ARMS:
        model.set_adapter(arm)
        for n,(row_id,enc) in enumerate(inputs,1):
            if (arm,row_id) in seen:
                continue
            vals={}
            for ch in ("chosen","rejected"):
                raw,avg,count=policy_completion_logps(
                    model,enc[ch+"_ids"],enc[ch+"_mask"],device=model.device)
                if count!=enc[ch+"_length"]:
                    raise ValueError("tokens scored differs from encoder")
                vals[ch+"_sum"]=raw
                vals[ch+"_avg"]=avg
                vals[ch+"_tokens"]=count
            obj={"arm":arm,"row_id":row_id,
                 "total_delta":vals["chosen_sum"]-vals["rejected_sum"],
                 "mean_delta":vals["chosen_avg"]-vals["rejected_avg"],
                 "chosen_tokens":vals["chosen_tokens"],
                 "rejected_tokens":vals["rejected_tokens"],
                 "chosen_truncated":enc["chosen_was_truncated"],
                 "rejected_truncated":enc["rejected_was_truncated"]}
            with process_file.open("a",encoding="utf-8") as out:
                out.write(json.dumps(obj,ensure_ascii=False)+"\n")
            seen[(arm,row_id)]=obj
            if n%20==0:
                print(f"TRAIN-007D {arm} scored {n}/{len(inputs)}",flush=True)
    paired=[]
    for row_id,enc in inputs:
        a=seen["sft",row_id]
        b=seen["dpo",row_id]
        paired.append({
            "row_id":row_id,
            "sft_mean_delta":a["mean_delta"],"dpo_mean_delta":b["mean_delta"],
            "sft_total_delta":a["total_delta"],"dpo_total_delta":b["total_delta"],
        })
    result=summarize(paired)
    result["frozen_inputs"]=provenance
    result["gpu"]={"device":torch.cuda.get_device_name(0),
                   "peak_reserved_bytes":torch.cuda.max_memory_reserved()}
    path=root/"summary.json"
    if path.exists():
        if read_json(path)!=result: raise ValueError("existing summary differs from completed evaluation")
    else:
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--root",type=Path,default=Path("artifacts/train007d-matched-preference"))
    a.add_argument("--sft-adapter",type=Path,default=Path("artifacts/posttrain-a/train007a-qlora-ultrachat"))
    a.add_argument("--dpo-adapter",type=Path,default=Path("artifacts/posttrain-a/train007c-dpo-ultrafeedback"))
    a.add_argument("--sft-summary",type=Path,default=Path("artifacts/posttrain-a/train007a-qlora-ultrachat/run_summary.json"))
    a.add_argument("--dpo-summary",type=Path,default=Path("artifacts/posttrain-a/train007c-dpo-ultrafeedback/run_summary.json"))
    a.add_argument("--heldout",type=Path,default=Path("data/generated/posttrain-h4-v1/dpo_validation.jsonl"))
    a.add_argument("--train",type=Path,default=Path("data/generated/posttrain-h4-v1/dpo_train.jsonl"))
    a.add_argument("--manifest",type=Path,default=Path("data/generated/posttrain-h4-v1/dpo_manifest.json"))
    args=a.parse_args()
    run(root=args.root,sft_adapter=args.sft_adapter,
        dpo_adapter=args.dpo_adapter,
        sft_summary=args.sft_summary,dpo_summary=args.dpo_summary,
        heldout_path=args.heldout,train_path=args.train,manifest_path=args.manifest)


if __name__=="__main__":
    main()
