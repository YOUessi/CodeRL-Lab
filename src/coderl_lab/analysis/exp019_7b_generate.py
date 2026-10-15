"""EXP-019: independently sourced MBPP+ frozen same-7B three-arm generation.

Generates 24 new-in-project heldout prompts * 4 draws * 2 seeds for each
SFT, previous GRPO16, and fresh GRPO32. Does not read heldout oracle. All
outputs are atomically append-checkpointed; frozen source/config/model
hashes guard resume. Scoring MUST wait until 576 outputs are frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from coderl_lab.datasets.exp019_mbppplus_freeze import (
    digest, EXPERIMENT, DATA_REVISION, SOURCE_SHA,
)
MODEL="Qwen/Qwen2.5-7B-Instruct"
REVISION="a09a35458c702b33eeacc393d103063234e8bc28"

ARMS=("sft","previous_grpo","new_grpo")
SEEDS=(42,314159)
N_TASKS=24
N_DRAWS=2
TOTAL=N_TASKS*N_DRAWS*len(SEEDS)*len(ARMS)
SFT_SHA="25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8"
PREV_SHA="3d84428201f8802cdcde33cb64dd2c46ee4f2538de850cb2240b017ff2a237c8"
MANIFEST_TRAIN_SHA="21a63fcef2e74d7e902480db9aabceb6b61d5f3bd4aff551e1959f7b8e38e4bb"
PROMPT_SHA="f6a8cb6082bfa624b471c6292416ed5de5dcaa1981001fa46558726162b1240b"
ORACLE_SHA="c36580dc3a99b962b98b261d521d023cea3fb3467031f51023e6a46dacac89f5"
GENERATION={
    "max_new_tokens":256,
    "temperature":0.8,
    "top_p":0.95,
    "do_sample":True,
    "base_quantization":"NF4_double_quant_BF16",
    "sampler":"same task, global seed and draw index for all three 7B adapters",
}


def samples_seed(task_id:str,seed:int,idx:int)->int:
    content=f"EXP019|{seed}|{task_id}|{idx}"
    return int(hashlib.sha256(content.encode("utf-8")).hexdigest()[:8],16)&0x7FFFFFFF


def read_jsonl(path:Path)->list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def verify_source(public:Path,manifest:Path,sft:Path,previous:Path,new:Path,
                  new_summary:Path,reference_summary:Path)->tuple[list[dict],dict]:
    m=json.loads(manifest.read_text(encoding="utf-8"))
    data=read_jsonl(public)
    trained=json.loads(new_summary.read_text(encoding="utf-8"))
    judge=json.loads(reference_summary.read_text(encoding="utf-8"))
    if (m.get("experiment")!=EXPERIMENT
        or m.get("dataset_revision")!=DATA_REVISION
        or m.get("original_parquet_sha256")!=SOURCE_SHA
        or m.get("selected_tasks")!=N_TASKS
        or m["sha256"]["public_prompts"]!=PROMPT_SHA
        or m["sha256"]["private_augmented_oracle"]!=ORACLE_SHA
        or m["evaluation_seeds"]!=list(SEEDS)
        or m["samples_per_task_per_seed_per_arm"]!=N_DRAWS
        or m["total_generations"]!=TOTAL
        or digest(public)!=PROMPT_SHA):
        raise ValueError("MBPP+ pinned data or generation protocol changed")
    if len(data)!=N_TASKS or len({r["task_id"] for r in data})!=N_TASKS:
        raise ValueError("24 unique public MBPP+ tasks required")
    if any(set(r)!={"task_id","prompt","source_task_id","public_examples_sha256"}
           or len(r["prompt"])!=1
           or r["prompt"][0].get("role")!="user"
           or "augmented_test" in str(r) or "reference_code" in str(r)
           for r in data):
        raise ValueError("private MBPP+ oracle/reference leaked to model generator")
    if (judge.get("status")!="completed" or judge.get("reference_pass")!=24
        or judge.get("tasks_tested")!=24 or judge.get("oracle_sha256")!=ORACLE_SHA
        or judge.get("no_candidate_model_outputs_scored") is not True):
        raise ValueError("MBPP+ reference smoke prerequisite not satisfied")
    for arm,path,expected in (("sft",sft,SFT_SHA),
                              ("previous_grpo",previous,PREV_SHA)):
        if digest(path/"adapter_model.safetensors")!=expected:
            raise ValueError(f"wrong frozen {arm} Adapter")
    recent_sha=digest(new/"adapter_model.safetensors")
    if (trained.get("status")!="real_GPU_GRPO_completed"
        or trained.get("global_step")!=32
        or trained.get("trainable_lora_dtype")!=["torch.float32"]
        or trained.get("warm_start_source_sft_adapter_sha256")!=SFT_SHA
        or trained.get("train_sha256")!=MANIFEST_TRAIN_SHA
        or trained.get("model_revision")!=REVISION
        or trained.get("saved_adapter_sha256")!=recent_sha):
        raise ValueError("TRAIN017 7B GRPO32 model weights not verified")
    return data,{"sft":SFT_SHA,"previous_grpo":PREV_SHA,"new_grpo":recent_sha}

def checkpoint_records(path:Path,valid_tasks:set[str])->dict:
    rows=read_jsonl(path) if path.exists() else []
    indexed={}
    for r in rows:
        key=(r["task_id"],r["seed"],r["sample_id"],r["arm"])
        if (key in indexed or key[0] not in valid_tasks or key[1] not in SEEDS
            or key[2] not in range(N_DRAWS) or key[3] not in ARMS
            or r["actual_seed"]!=samples_seed(*key[:3])
            or hashlib.sha256(r["answer"].encode()).hexdigest()!=r["answer_sha256"]):
            raise ValueError("duplicate/unknown/mutated checkpoint")
        indexed[key]=r
    return indexed


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--public",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--sft",type=Path,required=True)
    p.add_argument("--previous",type=Path,required=True)
    p.add_argument("--new",type=Path,required=True)
    p.add_argument("--new-summary",type=Path,required=True)
    p.add_argument("--reference-summary",type=Path,required=True)
    p.add_argument("--smoke-tasks",type=int)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    public_tasks,adapters=verify_source(
        args.public,args.manifest,args.sft,args.previous,args.new,
        args.new_summary,args.reference_summary
    )
    if args.smoke_tasks is not None and not 1<=args.smoke_tasks<=2:
        raise ValueError("smoke supports at most first2 tasks")
    tasks=public_tasks[:args.smoke_tasks] if args.smoke_tasks else public_tasks
    expected_count=len(tasks)*N_DRAWS*len(SEEDS)*len(ARMS)
    identity={
        "experiment":"EXP-019-MBPPPLUS-SMOKE" if args.smoke_tasks else "EXP-019-MBPPPLUS-GENERATION",
        "model":MODEL,"revision":REVISION,
        "arms_sha256":adapters,"public_prompt_sha256":PROMPT_SHA,
        "private_oracle_sha256_only":ORACLE_SHA,
        "oracle_reference_preflight_sha256":digest(args.reference_summary),
        "dataset_manifest_sha256":digest(args.manifest),
        "external_dataset_revision":DATA_REVISION,
        "new_training_summary_sha256":digest(args.new_summary),
        "source_script_sha256":digest(Path(__file__)),
        "seeds":list(SEEDS),"samples_per_task":N_DRAWS,
        "task_count":len(tasks),"expected_outputs":expected_count,
        "generation_config":GENERATION,
        "heldout_oracle_never_opened":True,
        "private_test_reference_code_never_opened":True,
        "historical_model_results_never_used_to_choose_parameters":True,
    }
    args.output_dir.mkdir(parents=True,exist_ok=True)
    idpath=args.output_dir/"identity.json"
    if idpath.exists():
        if json.loads(idpath.read_text())!=identity:
            raise ValueError("generation identity changed between runs")
    else:
        idpath.write_text(json.dumps(identity,ensure_ascii=False,indent=2)+"\n")
    checkpoint=args.output_dir/"generation_checkpoint.jsonl"
    done=checkpoint_records(checkpoint,{r["task_id"] for r in tasks})

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
    torch.cuda.reset_peak_memory_stats()
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,
                                      local_files_only=True,use_fast=True)
    quant=BitsAndBytesConfig(
        load_in_4bit=True,bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16,
    )
    backbone=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=REVISION,quantization_config=quant,
        local_files_only=True,dtype=torch.bfloat16,device_map={"":0},
    )
    model=PeftModel.from_pretrained(backbone,str(args.sft),adapter_name="sft",is_trainable=False)
    model.load_adapter(str(args.previous),adapter_name="previous_grpo",is_trainable=False)
    model.load_adapter(str(args.new),adapter_name="new_grpo",is_trainable=False)
    model.eval()
    model.config.use_cache=True
    adapter_dtype={
        arm:sorted({str(t.dtype) for k,t in model.named_parameters()
                    if ".lora_" in k and "."+arm+"." in k})
        for arm in ARMS
    }
    if any(adapter_dtype[arm]!=["torch.float32"] for arm in ARMS):
        raise ValueError("all inference LoRA adapters must be identically FP32")
    start=time.monotonic()
    with torch.inference_mode():
        for index,task in enumerate(tasks):
            prompt=tok.apply_chat_template(task["prompt"],tokenize=False,add_generation_prompt=True)
            ids=tok(prompt,return_tensors="pt").to(model.device)
            for seed in SEEDS:
                for draw in range(N_DRAWS):
                    actual_seed=samples_seed(task["task_id"],seed,draw)
                    for arm in ARMS:
                        key=(task["task_id"],seed,draw,arm)
                        if key in done:continue
                        torch.manual_seed(actual_seed)
                        torch.cuda.manual_seed_all(actual_seed)
                        model.set_adapter(arm)
                        out=model.generate(
                            **ids,do_sample=True,temperature=GENERATION["temperature"],
                            top_p=GENERATION["top_p"],max_new_tokens=GENERATION["max_new_tokens"],
                            use_cache=True,pad_token_id=tok.eos_token_id,
                        )
                        answer=tok.decode(out[0,ids["input_ids"].shape[1]:],
                                          skip_special_tokens=True).strip()
                        record={
                            "task_id":task["task_id"],"seed":seed,"sample_id":draw,
                            "actual_seed":actual_seed,"arm":arm,
                            "answer":answer,
                            "answer_sha256":hashlib.sha256(answer.encode()).hexdigest(),
                            "generated_tokens":int(out.shape[1]-ids["input_ids"].shape[1]),
                        }
                        with checkpoint.open("a",encoding="utf-8") as f:
                            f.write(json.dumps(record,ensure_ascii=False)+"\n");f.flush()
                        done[key]=record
            if (index+1)%3==0:
                print(f"EXP019 generation {index+1}/{len(tasks)} tasks {len(done)}/{expected_count} answers",flush=True)
    if len(done)!=expected_count:
        raise RuntimeError("not all fixed MBPP+ prompt-only model generations present")
    result={
        **identity,"status":"MBPPPLUS_GENERATION_FROZEN",
        "answers_generated":len(done),
        "generation_checkpoint_sha256":digest(checkpoint),
        "adapter_dtype":adapter_dtype,
        "generation_time_seconds":round(time.monotonic()-start,2),
        "gpu_peak_allocated_mib":round(torch.cuda.max_memory_allocated()/2**20,2),
        "oracle_labels_opened":False,
    }
    output=args.output_dir/"generation_summary.json"
    if output.exists():
        if json.loads(output.read_text())!=result:
            raise FileExistsError("existing generation summary differs")
    else:output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
