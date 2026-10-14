"""EXP-014: generate anonymous review outputs from prompt-only HH source.

This module MUST NOT read private_pairs.jsonl or reference labels. Its only
dataset input is the frozen 64 prompt-only messages. Three 7B adapters share
prompt, sampler, base weights, tokenization, and per-task random seeds.
Raw responses and blind-key remain on Tang; GitHub receives hashes only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from coderl_lab.datasets.exp014_hh_helpful import hash_file, MODEL, MODEL_REV

ARMS=("sft","bf16_dpo","fp32_dpo")
SHA={
    "sft":"25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8",
    "bf16_dpo":"d586d9ef32377189aa7d79b70ea5e37b5d5f26b930a4f60266ca024213f1a185",
    "fp32_dpo":"471703e1cec7fd322219133c10e09fa62aab2f6950b896688c7c073de1543281",
}
PUBLIC64_SHA="c58be29d6040c31e8a1340ca54f7056cf5ff9fc2bfb71ff901433b1114eae557"
CONFIG={
    "model":MODEL, "revision":MODEL_REV,
    "nf4":"double_quant_bf16_compute",
    "inference_adapter_dtype":"fp32_all_three",
    "temperature":0.7,"top_p":0.9,"max_new_tokens":224,
    "sampling":"one_draw_per_task_per_arm_seed42",
    "template":"qwen_chat_with_final_assistant_generation_prompt",
}


def seed(task_id:str)->int:
    h=hashlib.sha256(("EXP014-GEN|seed=42|"+task_id).encode()).digest()
    return int.from_bytes(h[:4],"little") % (2**31)


def output_sha(value:str)->str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_public(path:Path,manifest:Path)->list[dict]:
    x=json.loads(manifest.read_text(encoding="utf-8"))
    if x.get("experiment")!="EXP-014" or x.get("sha256",{}).get("generation_prompts")!=PUBLIC64_SHA:
        raise ValueError("prompt cohort identity changed")
    if hash_file(path)!=PUBLIC64_SHA:
        raise ValueError("generation prompt SHA drift")
    rows=[json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    if len(rows)!=64 or len({r["task_id"] for r in rows})!=64:
        raise ValueError("expected all64 public-only prompt IDs")
    for r in rows:
        if set(r)!={"task_id","messages","prompt_tokens","context_sha256"}:
            raise ValueError("labels or unknown data included in model generation")
        if not r["messages"] or r["messages"][-1]["role"]!="user":
            raise ValueError("last user message missing")
    return rows


def read_completed(path:Path,ids:set[str])->dict:
    records={}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():continue
            r=json.loads(line)
            key=(r["task_id"],r["arm"])
            if key in records or key[0] not in ids or key[1] not in ARMS:
                raise ValueError("duplicate/unknown completion checkpoint")
            if r["answer_sha256"]!=output_sha(r["answer"]) or r["seed"]!=seed(r["task_id"]):
                raise ValueError("checkpoint generator content/seed modified")
            records[key]=r
    return records


def blind_packet(rows:list[dict],records:dict,out:Path)->dict:
    packet_path=out/"blind_review_packets.jsonl"
    key_path=out/"blind_assignment_key.jsonl"
    if packet_path.exists() or key_path.exists():
        raise FileExistsError("blinded review packet already frozen")
    packed=[]
    keys=[]
    for item in rows:
        qid=item["task_id"]
        options=[{"arm":arm,"answer":records[qid,arm]["answer"]} for arm in ARMS]
        rng=random.Random(seed(qid)^0x13B17D)
        rng.shuffle(options)
        letters=["A","B","C"]
        packed.append({
            "task_id":qid, "messages":item["messages"],
            "candidate_answers":{letters[i]:options[i]["answer"] for i in range(3)},
            "instructions":"Rank A/B/C for helpfulness, factual correctness, clarity and relevance; tie permitted; leave comments. Models deliberately hidden.",
        })
        keys.append({"task_id":qid,
                     "mapping":{letters[i]:options[i]["arm"] for i in range(3)},
                     "answer_sha256":{letters[i]:output_sha(options[i]["answer"]) for i in range(3)}})
    for path,data in ((packet_path,packed),(key_path,keys)):
        with path.open("x",encoding="utf-8") as h:
            for item in data:h.write(json.dumps(item,ensure_ascii=False)+"\n")
    return {
        "blinded_packets":len(packed),
        "blinded_candidate_answers":len(packed)*3,
        "review_packet_sha256":hash_file(packet_path),
        "key_sha256":hash_file(key_path),
        "raw_text_uploaded":False,
        "human_review_status":"not_reviewed",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--public-prompts",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--sft",type=Path,required=True)
    p.add_argument("--bf16-dpo",type=Path,required=True)
    p.add_argument("--fp32-dpo",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--smoke-tasks",type=int)
    a=p.parse_args()
    all_rows=validate_public(a.public_prompts,a.manifest)
    if a.smoke_tasks is not None and not 1<=a.smoke_tasks<=4:
        raise ValueError("smoke cohort must be first1..4 tasks")
    selected=all_rows[:a.smoke_tasks] if a.smoke_tasks else all_rows
    for arm,path in (("sft",a.sft),("bf16_dpo",a.bf16_dpo),("fp32_dpo",a.fp32_dpo)):
        if hash_file(path/"adapter_model.safetensors")!=SHA[arm]:
            raise ValueError(f"source model adapter SHA changed: {arm}")
    a.output_dir.mkdir(parents=True,exist_ok=True)
    identity={
        "experiment":"EXP-014-HH-BLIND-GENERATION",
        "config":CONFIG,"data_public_sha256":PUBLIC64_SHA,
        "generation_source_sha256":hash_file(Path(__file__)),
        "arms_sha256":SHA,
        "num_prompts":len(selected),"num_outputs":len(selected)*3,
        "selection":"frozen_first_n_smoke" if a.smoke_tasks else "frozen_full64",
        "reference_labels_never_accessed":True,
    }
    stamp=a.output_dir/"identity.json"
    if stamp.exists():
        if json.loads(stamp.read_text())!=identity:
            raise ValueError("generation checkpoint identity/config mismatch")
    else:
        stamp.write_text(json.dumps(identity,indent=2)+"\n")
    checkpoint=a.output_dir/"generation_checkpoint.jsonl"
    completed=read_completed(checkpoint,{r["task_id"] for r in selected})
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
    from peft import PeftModel
    tokenizer=AutoTokenizer.from_pretrained(
        MODEL,revision=MODEL_REV,local_files_only=True,use_fast=True)
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
       bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    base=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=MODEL_REV,local_files_only=True,device_map={"":0},
        quantization_config=q,dtype=torch.bfloat16)
    model=PeftModel.from_pretrained(base,str(a.sft),adapter_name="sft",is_trainable=False)
    model.load_adapter(str(a.bf16_dpo),adapter_name="bf16_dpo",is_trainable=False)
    model.load_adapter(str(a.fp32_dpo),adapter_name="fp32_dpo",is_trainable=False)
    model.eval()
    dtypes={
        arm: sorted({str(p.dtype) for n,p in model.named_parameters()
                    if ".lora_" in n and "."+arm+"." in n})
        for arm in ARMS
    }
    if any(dtypes[arm]!=["torch.float32"] for arm in ARMS):
        raise ValueError("all models must have identical FP32 inference LoRA dtype")
    start=time.monotonic()
    for i,row in enumerate(selected,1):
        qid=row["task_id"]
        prompt=tokenizer.apply_chat_template(
            row["messages"],tokenize=False,add_generation_prompt=True)
        tokens=tokenizer(prompt,return_tensors="pt").to(model.device)
        for arm in ARMS:
            key=(qid,arm)
            if key in completed:continue
            torch.manual_seed(seed(qid))
            torch.cuda.manual_seed_all(seed(qid))
            model.set_adapter(arm)
            with torch.inference_mode():
                output=model.generate(
                    **tokens,temperature=.7,top_p=.9,max_new_tokens=224,
                    do_sample=True,pad_token_id=tokenizer.eos_token_id,
                )
            ans=tokenizer.decode(
                output[0,tokens["input_ids"].shape[1]:],skip_special_tokens=True).strip()
            record={
                "task_id":qid,"arm":arm,"seed":seed(qid),
                "answer":ans,"answer_sha256":output_sha(ans),
                "generated_tokens":int(output.shape[1]-tokens["input_ids"].shape[1]),
                "context_sha256":row["context_sha256"],
            }
            with checkpoint.open("a",encoding="utf-8") as h:
                h.write(json.dumps(record,ensure_ascii=False)+"\n")
                h.flush()
            completed[key]=record
        if i%8==0 or i==len(selected):
            print(f"EXP014 BLIND GENERATION {i}/{len(selected)} tasks, {len(completed)}/{len(selected)*3} answers",flush=True)
    if len(completed)!=len(selected)*3:
        raise RuntimeError("not all 3-arm answers generated")
    packet=blind_packet(selected,completed,a.output_dir)
    summary={
        **identity,"status":"generated_blinded_needs_human_review",
        "answers_generated":len(completed),
        "generation_checkpoint_sha256":hash_file(checkpoint),
        "source_prompts_sha256":hash_file(a.public_prompts),
        "inference_lora_dtypes":dtypes,
        "generated_tokens":{
            arm:sum(r["generated_tokens"] for r in completed.values() if r["arm"]==arm)
            for arm in ARMS},
        "peak_allocated_cuda_mib":round(torch.cuda.max_memory_allocated()/2**20,2),
        "elapsed_seconds":round(time.monotonic()-start,2),
        **packet,
    }
    (a.output_dir/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":
    main()
