"""TRAIN-017 frozen 576-result Docker evaluator (no model generation/training).

Scores the preregistered 24 heldout task oracles only AFTER all 576
three-arm/two-seed generations are immutable. Uses actual pinned Docker
sandbox. Cluster bootstrap by TASK, not by repeated seeds/completions.
Never accesses future training labels or runs generated code on the host.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

from coderl_lab.datasets.train017_new_algorithms import digest
from coderl_lab.analysis.train017_7b_generate import (
    ARMS,SEEDS,N_TASKS,N_DRAWS,TOTAL,PROMPT_SHA,ORACLE_SHA,
    checkpoint_records,read_jsonl,
)
from coderl_lab.train.train014b_reward_sandbox import (
    check_pinned_docker,score_one,
)

def paired_ci(delta:list[float],reps:int=20000,seed:int=42)->dict:
    if len(delta)!=N_TASKS:
        raise ValueError("bootstrap must cluster 24 tasks")
    rng=random.Random(seed)
    sorted_estimates=sorted(
        sum(delta[rng.randrange(len(delta))] for _ in range(len(delta)))/len(delta)
        for _ in range(reps)
    )
    return {
        "estimate":sum(delta)/len(delta),
        "ci95_low":sorted_estimates[int(.025*reps)],
        "ci95_high":sorted_estimates[int(.975*reps)-1],
        "iterations":reps,"seed":seed,"resample_unit":"task_24_not_repeated_seeds",
    }


def scored_rows_valid(scored:dict,generated:dict)->None:
    for key,row in scored.items():
        if (key not in generated
            or row["answer_sha256"]!=generated[key]["answer_sha256"]
            or type(row.get("all_passed")) is not bool
            or not 0<=row.get("reward",-1)<=1):
            raise ValueError("scored verdict mismatch/invalid result")


def group_scores(tasks:list[dict],scored:dict,generated:dict)->dict:
    metrics={arm:{
        "candidate_correct":0,"all_candidates":N_TASKS*N_DRAWS*len(SEEDS),
        "sample0_correct":0,"sample0_total":N_TASKS*len(SEEDS),
        "solved4":0,"solved4_total":N_TASKS*len(SEEDS),
        "reward_total":0.0,"python_valid":0,"entry_valid":0,
        "cap256_hits":0,"verdict_histogram":Counter(),
        "by_seed":{str(seed):{"sample0_correct":0,"solved4":0,"correct_candidates":0} for seed in SEEDS}
    } for arm in ARMS}
    per_task=[]
    for task in tasks:
        qid=task["task_id"]
        task_summary={"task_id_hash":hashlib.sha256(qid.encode()).hexdigest()}
        for arm in ARMS:
            per_seed=[]
            for seed in SEEDS:
                batch=[scored[qid,seed,n,arm] for n in range(N_DRAWS)]
                success=[int(t["all_passed"]) for t in batch]
                label=str(seed)
                info=metrics[arm]
                info["candidate_correct"]+=sum(success)
                info["sample0_correct"]+=success[0]
                info["solved4"]+=int(any(success))
                info["by_seed"][label]["sample0_correct"]+=success[0]
                info["by_seed"][label]["solved4"]+=int(any(success))
                info["by_seed"][label]["correct_candidates"]+=sum(success)
                info["reward_total"]+=sum(t["reward"] for t in batch)
                info["python_valid"]+=sum(t["python_valid"] for t in batch)
                info["entry_valid"]+=sum(t["entry_valid"] for t in batch)
                info["cap256_hits"]+=sum(
                    int(generated[qid,seed,n,arm]["generated_tokens"]>=256)
                    for n in range(N_DRAWS)
                )
                info["verdict_histogram"].update(t["verdict"] for t in batch)
                per_seed.append({
                    "at1":success[0],
                    "at4":int(any(success)),
                    "correct_fraction":sum(success)/N_DRAWS,
                    "reward_mean":sum(t["reward"] for t in batch)/N_DRAWS,
                })
            task_summary[arm]={
                metric:sum(x[metric] for x in per_seed)/len(SEEDS)
                for metric in ("at1","at4","correct_fraction","reward_mean")
            }
        per_task.append(task_summary)
    for arm in ARMS:
        m=metrics[arm]
        m["candidate_accuracy"]=m["candidate_correct"]/m["all_candidates"]
        m["pass_at1_frequency"]=m["sample0_correct"]/m["sample0_total"]
        m["solved_at4_frequency"]=m["solved4"]/m["solved4_total"]
        m["mean_partial_reward"]=m.pop("reward_total")/m["all_candidates"]
        m["verdict_histogram"]=dict(sorted(m["verdict_histogram"].items()))
    comparisons={}
    for first,second in (("new_grpo","sft"),("previous_grpo","sft"),("new_grpo","previous_grpo")):
        key=f"{first}_minus_{second}"
        comparisons[key]={
            unit:paired_ci([t[first][unit]-t[second][unit] for t in per_task])
            for unit in ("at1","at4","correct_fraction","reward_mean")
        }
    return {
        "status":"completed_all576_docker_scored",
        "task_count":N_TASKS,"task_seed_pairs":N_TASKS*len(SEEDS),
        "model_arms":list(ARMS),
        "samples_each_arm":N_TASKS*N_DRAWS*len(SEEDS),
        "total_candidates":TOTAL,
        "seeds":list(SEEDS),
        "metrics":metrics,
        "pairwise_task_cluster_bootstrap":comparisons,
        "note":"solved_at4 is observed at-least-once correctness across 4 draws, not pass@4 unbiased estimator. The two seeds are clustered within each of 24 tasks in CI.",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--public",type=Path,required=True)
    p.add_argument("--oracle",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--generation-dir",type=Path,required=True)
    p.add_argument("--workers",type=int,default=4)
    args=p.parse_args()
    if not 1<=args.workers<=4:raise ValueError("bounded Docker parallelism 1..4")
    m=json.loads(args.manifest.read_text())
    generation_summary_path=args.generation_dir/"generation_summary.json"
    generation_summary=json.loads(generation_summary_path.read_text())
    identity=json.loads((args.generation_dir/"identity.json").read_text())
    checkpoint=args.generation_dir/"generation_checkpoint.jsonl"
    if (generation_summary.get("status")!="all_576_generated_frozen"
        or generation_summary.get("answers_generated")!=TOTAL
        or generation_summary.get("generation_checkpoint_sha256")!=digest(checkpoint)
        or generation_summary.get("heldout_oracle_never_opened") is not True
        or identity.get("expected_outputs")!=TOTAL
        or identity.get("private_oracle_sha256_only")!=ORACLE_SHA
        or m["sha256"]["holdout_public"]!=digest(args.public)
        or m["sha256"]["holdout_oracle"]!=ORACLE_SHA):
        raise ValueError("generations not fully frozen BEFORE oracle access")
    public=read_jsonl(args.public)
    if len(public)!=N_TASKS or len({r["task_id"] for r in public})!=N_TASKS:
        raise ValueError("24 public prompt tasks required")
    done=checkpoint_records(checkpoint,{r["task_id"] for r in public})
    if len(done)!=TOTAL:
        raise ValueError("incomplete 576 output matrix before oracle")
    # This is the FIRST code path that parses private heldout expected results.
    if digest(args.oracle)!=ORACLE_SHA:
        raise ValueError("heldout private oracle SHA changed")
    oracle=read_jsonl(args.oracle)
    if len(oracle)!=N_TASKS or {r["task_id"] for r in oracle}!={r["task_id"] for r in public}:
        raise ValueError("heldout reference oracle task IDs differ")
    tests={r["task_id"]:r["test_cases"] for r in oracle}
    check_pinned_docker()
    sc_path=args.generation_dir/"docker_scored_checkpoint.jsonl"
    scored={}
    if sc_path.exists():
        for row in read_jsonl(sc_path):
            key=(row["task_id"],row["seed"],row["sample_id"],row["arm"])
            if key in scored:
                raise ValueError("duplicate Docker checkpoint entry")
            scored[key]=row
        scored_rows_valid(scored,done)
    absent=[key for key in sorted(done) if key not in scored]
    print(f"TRAIN017 Docker scorer starting: already={len(scored)} remaining={len(absent)}",flush=True)

    def evaluate(key):
        sample=done[key]
        outcome=score_one(sample["answer"],tests[key[0]])
        return key,{
            "task_id":key[0],"seed":key[1],
            "sample_id":key[2],"arm":key[3],
            "answer_sha256":sample["answer_sha256"],
            "all_passed":bool(outcome["all_passed"]),
            "reward":float(outcome["reward"]),
            "python_valid":bool(outcome["python_valid"]),
            "entry_valid":bool(outcome["entry_valid"]),
            "verdict":outcome["verdict"],
            "normalized_code_sha256":outcome["code_sha256"],
        }
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(evaluate,key):key for key in absent}
        for fut in as_completed(futures):
            key,row=fut.result()
            if key in scored:raise ValueError("duplicate completed scorer result")
            with sc_path.open("a",encoding="utf-8") as f:
                f.write(json.dumps(row,ensure_ascii=False)+"\n");f.flush()
            scored[key]=row
            if len(scored)%48==0:
                print(f"TRAIN017 real Docker verdicts {len(scored)}/{TOTAL}",flush=True)
    if len(scored)!=TOTAL or set(scored)!=set(done):
        raise RuntimeError("missing one of 576 Docker verdicts")
    scored_rows_valid(scored,done)
    outcome=group_scores(public,scored,done)
    result={
        "experiment":"TRAIN-017-7B-FRESH-HOLDOUT-THREE-ARM",
        **outcome,
        "generation_summary_sha256":digest(generation_summary_path),
        "frozen_outputs_sha256":digest(checkpoint),
        "frozen_oracle_sha256":ORACLE_SHA,
        "docker_scored_checkpoint_sha256":digest(sc_path),
        "Docker_image_sha256":"sha256:d3846bce6e97d0ef5c32fc023300d5e39251ad791e6fdfaee41d2ba75246766e",
        "model_weight_sha256":identity["arms_sha256"],
        "first_read_oracle_after_generation_frozen":True,
        "all_generated_programs_executed_only_in_isolated_Docker":True,
        "limitations":"These are 24 deterministic synthetic algorithmic tasks newly constructed in this project, split by family parameter; possible pretrained familiarity remains. Multiple seeds are correlated within task. Not a guarantee of broad real-world coding improvement.",
    }
    path=args.generation_dir/"docker_score_summary.json"
    if path.exists():
        if json.loads(path.read_text())!=result:
            raise FileExistsError("refusing to overwrite different Docker summary")
    else:
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
