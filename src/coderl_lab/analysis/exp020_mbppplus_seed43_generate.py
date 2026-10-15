"""EXP-019B: identical Qwen2.5-7B SFT/GRPO32 MBPP+ answer generation.

Generation consumes ONLY 24 public MBPP+ prompts and function signatures,
NEVER private tests or reference implementation. Freeze all 24*4*2=192
source-code completions by SHA before the scorer can read hidden tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from coderl_lab.datasets.exp019b_mbppplus_freeze import digest,SOURCE_SHA,COHORT

MODEL="Qwen/Qwen2.5-7B-Instruct"
REVISION="a09a35458c702b33eeacc393d103063234e8bc28"
SFT_SHA="25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8"
GRPO_SHA="02ff43a2aa674c58d25ae4691e7f2269689851fb17037282e5d89a82871d4aee"
ARMS=("sft","grpo")
DRAWS=4
CONFIG={"max_new_tokens":256,"temperature":0.8,"top_p":0.95,"do_sample":True,
        "base_quantization":"NF4_double_quant_BF16","seed":42,
        "scope":"24 fixed MBPP+ test problems, original prompt + function signature"}


def sample_seed(task_id:str,sample_id:int)->int:
    key=f"EXP019B|42|{task_id}|{sample_id}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8],16)&0x7fffffff


def read_jsonl(path:Path)->list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def validate_public(prompts:Path,manifest:Path)->tuple[list[dict],dict]:
    m=json.loads(manifest.read_text())
    data=read_jsonl(prompts)
    if (m.get("experiment")!="EXP-019B-MBPPPLUS-EXTERNAL"
        or m.get("original_source_sha256")!=SOURCE_SHA
        or m.get("public_prompt_sha256")!=digest(prompts)
        or m.get("selected_task_count")!=COHORT
        or len(data)!=COHORT or len({p["task_id"] for p in data})!=COHORT):
        raise ValueError("MBPP+ public prompt cohort changed")
    for row in data:
        if (set(row)!={"task_id","source_task_id","entry_point","signature","description","messages"}
            or len(row["messages"])!=1 or row["messages"][0]["role"]!="user"
            or any(secret in row for secret in ("code","test","canonical_code","test_list"))
            or row["entry_point"] not in row["signature"]):
            raise ValueError("hidden MBPP+ answer/test leaked into public-only generator")
    return data,m


def read_checkpoint(path:Path,public:list[dict])->dict:
    accepted={r["task_id"] for r in public}
    seen={}
    if path.exists():
        for row in read_jsonl(path):
            key=(row["task_id"],row["sample_id"],row["arm"])
            if (key in seen or row["task_id"] not in accepted
                or row["sample_id"] not in range(DRAWS) or row["arm"] not in ARMS
                or row["seed"]!=sample_seed(row["task_id"],row["sample_id"])
                or row["code_sha256"]!=hashlib.sha256(row["raw_code"].encode()).hexdigest()):
                raise ValueError("generation checkpoint duplicate/inconsistent")
            seen[key]=row
    return seen


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--public",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--sft",type=Path,required=True)
    p.add_argument("--grpo",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--smoke-tasks",type=int,default=0)
    args=p.parse_args()
    public,m=validate_public(args.public,args.manifest)
    if args.smoke_tasks not in (0,2):
        raise ValueError("smoke only fixed first2 tasks, full runs all24")
    if args.smoke_tasks:public=public[:args.smoke_tasks]
    for pth,sha in ((args.sft,SFT_SHA),(args.grpo,GRPO_SHA)):
        if digest(pth/"adapter_model.safetensors")!=sha:
            raise ValueError("changed 7B LoRA adapter source")
    args.out.mkdir(parents=True,exist_ok=True)
    identity={
        "experiment":"EXP-019B-7B-SFT-VS-GRPO-MBPPPLUS",
        "source_dataset_sha256":SOURCE_SHA,
        "public_sha256":digest(args.public),
        "private_oracle_sha256_only":m["private_test_sha256"],
        "source_script_sha256":digest(Path(__file__)),
        "model_revision":REVISION,
        "model_adapters_sha256":{"sft":SFT_SHA,"grpo":GRPO_SHA},
        "configuration":CONFIG,
        "num_tasks":len(public),
        "expected_output_count":len(public)*DRAWS*len(ARMS),
        "sample_ids":list(range(DRAWS)),
        "data_selection":"SHA256 EXP019B original task_id, frozen before generation",
        "private_tests_never_opened":True,
        "generation_cannot_use_canonical_answers":True,
        "stage":"smoke2" if args.smoke_tasks else "full24",
    }
    frozen=args.out/"identity.json"
    if frozen.exists():
        if json.loads(frozen.read_text())!=identity:
            raise ValueError("generation identity or source code drift")
    else:frozen.write_text(json.dumps(identity,ensure_ascii=False,indent=2)+"\n")
    checkpoint=args.out/"checkpoint.jsonl"
    completed=read_checkpoint(checkpoint,public)
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    from peft import PeftModel
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True,use_fast=True)
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    base=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=REVISION,local_files_only=True,
        quantization_config=quant,dtype=torch.bfloat16,device_map={"":0})
    model=PeftModel.from_pretrained(base,str(args.sft),adapter_name="sft",is_trainable=False)
    model.load_adapter(str(args.grpo),adapter_name="grpo",is_trainable=False)
    model.eval()
    model.config.use_cache=True
    types={
        arm:sorted({str(param.dtype) for name,param in model.named_parameters()
                    if ".lora_" in name and "."+arm+"." in name})
        for arm in ARMS
    }
    if any(types[a]!=["torch.float32"] for a in ARMS):
        raise ValueError("model adapter inference precision not matched")
    torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    with torch.inference_mode():
        for task_index,row in enumerate(public,1):
            prompt=tok.apply_chat_template(
                row["messages"],tokenize=False,add_generation_prompt=True)
            tokens=tok(prompt,return_tensors="pt").to(model.device)
            for sample in range(DRAWS):
                n=sample_seed(row["task_id"],sample)
                for arm in ARMS:
                    key=(row["task_id"],sample,arm)
                    if key in completed:continue
                    torch.manual_seed(n);torch.cuda.manual_seed_all(n)
                    model.set_adapter(arm)
                    y=model.generate(
                        **tokens,do_sample=True,
                        temperature=CONFIG["temperature"],top_p=CONFIG["top_p"],
                        max_new_tokens=CONFIG["max_new_tokens"],use_cache=True,
                        pad_token_id=tok.eos_token_id)
                    answer=tok.decode(
                        y[0,tokens["input_ids"].shape[1]:],skip_special_tokens=True).strip()
                    saved={
                        "task_id":row["task_id"],
                        "sample_id":sample,"arm":arm,"seed":n,
                        "raw_code":answer,
                        "code_sha256":hashlib.sha256(answer.encode()).hexdigest(),
                        "generated_tokens":int(y.shape[1]-tokens["input_ids"].shape[1]),
                    }
                    with checkpoint.open("a",encoding="utf-8") as f:
                        f.write(json.dumps(saved,ensure_ascii=False)+"\n");f.flush()
                    completed[key]=saved
            if task_index%2==0:
                print(f"EXP019B 7B generated {task_index}/{len(public)} prompts; {len(completed)}/{identity['expected_output_count']} programs",flush=True)
    if len(completed)!=identity["expected_output_count"]:
        raise RuntimeError("missing generated SFT/GRPO samples")
    report={
        **identity,
        "status":"all_samples_generated_frozen",
        "answers_generated":len(completed),
        "checkpoint_sha256":digest(checkpoint),
        "model_inference_adapter_dtype":types,
        "gpu_peak_allocated_mib":round(torch.cuda.max_memory_allocated()/2**20,2),
        "time_seconds":round(time.monotonic()-started,2),
        "private_mbppplus_test_labels_opened":False,
        "raw_model_answers_exported_to_GitHub":False,
    }
    output=args.out/"generation_summary.json"
    if output.exists():
        if json.loads(output.read_text())!=report:
            raise FileExistsError("refusing generation summary replacement")
    else:output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
