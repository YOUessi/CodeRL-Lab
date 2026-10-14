"""TRAIN-014B frozen 12-task paired baseline-vs-GRPO code generalization.

Phase generate: GPU, reading ONLY public heldout prompts; never opens tests.
Phase score: after all 12*4*2 completions frozen, Docker executes heldout
tests and exports only aggregated pass@1/pass@4, reward and uncertainty.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import random
import time
from pathlib import Path

from coderl_lab.datasets.train014b_algorithmic_tasks import MODEL,REVISION,sha
from coderl_lab.train.train014b_reward_sandbox import (
    IMAGE_ID,check_pinned_docker,score_one,
)

ARMS=("base","grpo")
TASKS=12
SAMPLES=4
SEED=42


def rows(path:Path)->list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def sample_seed(task:str,sample:int)->int:
    return int(hashlib.sha256(f"TRAIN014B|42|{task}|{sample}".encode()).hexdigest()[:8],16)&0x7FFFFFFF


def frozen_state(public:Path,manifest:Path,summary:Path,adapter:Path)->dict:
    source=json.loads(manifest.read_text())
    trained=json.loads(summary.read_text())
    if (source.get("experiment")!="TRAIN-014B-ALGORITHMIC-GRPO"
        or source.get("heldout_oracle_never_seen_by_GRPOTrainer") is not True
        or source["sha256"]["holdout_public"]!=sha(public)
        or trained.get("status")!="real_GPU_GRPO_completed"
        or trained.get("global_step")!=16
        or trained.get("trainable_parameters",0)<=0
        or trained.get("reward_group_variation_tasks",0)<=0
        or trained["model_revision"]!=REVISION
        or sha(adapter/"adapter_model.safetensors")!=trained["saved_adapter_sha256"]):
        raise ValueError("no fully trained and verified 16-step GRPO LoRA checkpoint")
    prompts=rows(public)
    if len(prompts)!=TASKS or len({r["task_id"] for r in prompts})!=TASKS:
        raise ValueError("not exactly12 independent heldout prompts")
    if any(set(r)!={"task_id","prompt","task_kind"} or "test_cases" in r for r in prompts):
        raise ValueError("heldout oracle exposed during generation")
    return {
        "experiment":"TRAIN-014B-HOLDOUT",
        "model":MODEL,"revision":REVISION,
        "heldout_public_sha256":sha(public),
        "heldout_oracle_sha256":source["sha256"]["holdout_oracle"],
        "train_summary_sha256":sha(summary),
        "GRPO_adapter_sha256":trained["saved_adapter_sha256"],
        "reward_variation_training_tasks":trained["reward_group_variation_tasks"],
        "num_tasks":TASKS,"samples_per_task":SAMPLES,
        "model_arms":list(ARMS),
        "seed":SEED,
        "max_new_tokens":192,"temperature":.8,"top_p":.95,
        "baseline_and_grpo_same_prompt_sampler_and_backbone":True,
        "scores_not_seen_during_generation":True,
    }


def verify_checkpoint(path:Path,all_tasks:list[dict])->dict:
    known={r["task_id"] for r in all_tasks}
    done={}
    if path.exists():
        for record in rows(path):
            key=(record["task_id"],record["arm"],int(record["sample_id"]))
            if key in done or key[0] not in known or key[1] not in ARMS or key[2] not in range(SAMPLES):
                raise ValueError("unknown/duplicated completion")
            if (record["seed"]!=sample_seed(record["task_id"],record["sample_id"])
                or hashlib.sha256(record["raw_code"].encode()).hexdigest()!=record["completion_sha256"]):
                raise ValueError("saved completion code/seed changed")
            done[key]=record
    return done


def generate(a):
    manifest=json.loads(a.manifest.read_text())
    identity=frozen_state(a.public,a.manifest,a.training_summary,a.adapter)
    identity["generator_source_sha256"]=sha(Path(__file__))
    a.output_dir.mkdir(parents=True,exist_ok=True)
    path=a.output_dir/"identity.json"
    if path.exists():
        if json.loads(path.read_text())!=identity:raise ValueError("frozen generated identity changed")
    else:path.write_text(json.dumps(identity,indent=2)+"\n")
    public=rows(a.public)
    checkpoint=a.output_dir/"generations.jsonl"
    done=verify_checkpoint(checkpoint,public)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM,AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=REVISION,local_files_only=True,
        dtype=torch.bfloat16,device_map={"":0})
    model=PeftModel.from_pretrained(base,str(a.adapter),adapter_name="grpo",is_trainable=False)
    model.eval()
    started=time.monotonic()
    with torch.inference_mode():
        for i,task in enumerate(public,1):
            prompt=tokenizer.apply_chat_template(task["prompt"],tokenize=False,add_generation_prompt=True)
            enc=tokenizer(prompt,return_tensors="pt").to(model.device)
            for sample in range(SAMPLES):
                seed=sample_seed(task["task_id"],sample)
                for arm in ARMS:
                    key=(task["task_id"],arm,sample)
                    if key in done:continue
                    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                    enabled=model.disable_adapter() if arm=="base" else contextlib.nullcontext()
                    with enabled:
                        token_ids=model.generate(
                            **enc,do_sample=True,temperature=.8,top_p=.95,
                            max_new_tokens=192,pad_token_id=tokenizer.eos_token_id,
                        )
                    raw=tokenizer.decode(
                        token_ids[0,enc["input_ids"].shape[1]:],
                        skip_special_tokens=True).strip()
                    record={
                        "task_id":task["task_id"],"arm":arm,
                        "sample_id":sample,"seed":seed,
                        "raw_code":raw,
                        "completion_sha256":hashlib.sha256(raw.encode()).hexdigest(),
                        "tokens":int(token_ids.shape[1]-enc["input_ids"].shape[1]),
                    }
                    with checkpoint.open("a",encoding="utf-8") as f:
                        f.write(json.dumps(record,ensure_ascii=False)+"\n");f.flush()
                    done[key]=record
            if i%2==0:
                print(f"TRAIN014B heldout generation {i}/{TASKS} tasks {len(done)}/96",flush=True)
    if len(done)!=TASKS*SAMPLES*2:
        raise RuntimeError("missing baseline/grpo samples")
    report={**identity,"status":"generated_frozen_96",
            "model_generation_count":len(done),
            "generation_checkpoint_sha256":sha(checkpoint),
            "GPU_peak_allocated_mib":round(torch.cuda.max_memory_allocated()/2**20,2),
            "elapsed_seconds":round(time.monotonic()-started,2)}
    (a.output_dir/"generation_summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


def ci(values:list[float],num:int=20000)->dict:
    rng=random.Random(SEED);n=len(values)
    v=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(num))
    return {"estimate":sum(values)/n,"ci95_low":v[int(.025*num)],
            "ci95_high":v[int(.975*num)-1],"iterations":num,"seed":SEED}


def score(a):
    if sha(a.oracle)!=json.loads(a.manifest.read_text())["sha256"]["holdout_oracle"]:
        raise ValueError("heldout oracle changed")
    generated=json.loads((a.output_dir/"generation_summary.json").read_text())
    identity=json.loads((a.output_dir/"identity.json").read_text())
    checkpoint=a.output_dir/"generations.jsonl"
    if generated["status"]!="generated_frozen_96" or sha(checkpoint)!=generated["generation_checkpoint_sha256"]:
        raise ValueError("generated codes not fully frozen before oracle access")
    public=rows(a.public)
    done=verify_checkpoint(checkpoint,public)
    if len(done)!=96:
        raise ValueError("not exactly96 frozen model completions")
    # At this point, and not during generation, may the heldout oracle be opened.
    oracle=rows(a.oracle)
    if len(oracle)!=TASKS or {r["task_id"] for r in oracle}!={r["task_id"] for r in public}:
        raise ValueError("heldout oracle IDs mismatch")
    by_task={r["task_id"]:r["test_cases"] for r in oracle}
    check_pinned_docker()
    output=a.output_dir/"scored_checkpoint.jsonl"
    scored={}
    if output.exists():
        for r in rows(output):
            key=(r["task_id"],r["arm"],int(r["sample_id"]))
            if key in scored or key not in done or r["completion_sha256"]!=done[key]["completion_sha256"]:
                raise ValueError("partial scoring output drift")
            scored[key]=r
    for i,key in enumerate(sorted(done,key=lambda t:(t[0],t[2],t[1])),1):
        if key in scored:continue
        generated=done[key]
        result=score_one(generated["raw_code"],by_task[key[0]])
        saved={
            "task_id":key[0],"arm":key[1],"sample_id":key[2],
            "completion_sha256":generated["completion_sha256"],
            "all_passed":bool(result["all_passed"]),
            "partial_reward":float(result["reward"]),
            "verdict":result["verdict"],
        }
        with output.open("a",encoding="utf-8") as f:
            f.write(json.dumps(saved)+"\n");f.flush()
        scored[key]=saved
        if i%16==0:print(f"TRAIN014B Docker scored {i}/96",flush=True)
    per_task=[]
    for p in public:
        q=p["task_id"]
        a0=[scored[q,"base",n]["all_passed"] for n in range(SAMPLES)]
        a1=[scored[q,"grpo",n]["all_passed"] for n in range(SAMPLES)]
        per_task.append({
            "task_id":q,
            "baseline_solved_at1":a0[0],
            "grpo_solved_at1":a1[0],
            "baseline_solved_at4":any(a0),
            "grpo_solved_at4":any(a1),
            "baseline_correct_completions":sum(a0),
            "grpo_correct_completions":sum(a1),
        })
    totals={
        arm:{
            "candidate_correct":sum(scored[q,arm,n]["all_passed"] for q in by_task for n in range(SAMPLES)),
            "solved_at1":sum(t[f"{'baseline' if arm=='base' else 'grpo'}_solved_at1"] for t in per_task),
            "solved_at4":sum(t[f"{'baseline' if arm=='base' else 'grpo'}_solved_at4"] for t in per_task),
            "partial_reward_mean":sum(scored[q,arm,n]["partial_reward"] for q in by_task for n in range(SAMPLES))/48,
        } for arm in ARMS
    }
    summary={
        **identity,"status":"completed_heldout_baseline_grpo",
        "tasks":TASKS,"candidates":96,
        "scores_by_arm":totals,
        "paired_task_bootstrap":{
            "pass_at1_diff":ci([int(t["grpo_solved_at1"])-int(t["baseline_solved_at1"]) for t in per_task]),
            "pass_at4_diff":ci([int(t["grpo_solved_at4"])-int(t["baseline_solved_at4"]) for t in per_task]),
            "candidate_correct_fraction_diff":ci([(t["grpo_correct_completions"]-t["baseline_correct_completions"])/4 for t in per_task]),
        },
        "score_checkpoint_sha256":sha(output),
        "generated_checkpoint_sha256":sha(checkpoint),
        "source_oracle_sha256":sha(a.oracle),
        "disclaimer":"Small synthetic holdout of12 tasks; cannot establish broad code capability gains. All completion code executed only inside pinned Docker, not on host.",
    }
    (a.output_dir/"score_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",choices=["generate","score"],required=True)
    p.add_argument("--public",type=Path,required=True)
    p.add_argument("--oracle",type=Path)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--training-summary",type=Path)
    p.add_argument("--adapter",type=Path)
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args()
    if a.phase=="generate":
        if not a.training_summary or not a.adapter:
            raise ValueError("frozen GRPO model required to generate")
        generate(a)
    else:
        if a.oracle is None:raise ValueError("test oracle required only in score phase")
        score(a)

if __name__=="__main__":main()
