"""EXP-019 MBPP+ externally sourced strict hidden-augmented Docker scorer.

Only after all 288 three-7B generations are frozen/hashed may this module
open the private MBPP+ augmented test oracle. It never executes candidate
code on Tang host. Separate model arms share one pinned NumPy-enabled image.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

from coderl_lab.datasets.exp019_mbppplus_freeze import digest, EXPERIMENT, DATA_REVISION
from coderl_lab.analysis.exp019_7b_generate import (
    ARMS,SEEDS,N_TASKS,N_DRAWS,TOTAL,PROMPT_SHA,ORACLE_SHA,
    checkpoint_records,read_jsonl,
)
from coderl_lab.analysis.exp019_mbpp_docker import image_sha,test_program,IMAGE_ID
from coderl_lab.train.train014b_reward_sandbox import normalize_code


def ci(values:list[float],n:int=20000,seed:int=42)->dict:
    if len(values)!=N_TASKS:
        raise ValueError("must bootstrap 24 distinct tasks, not two seeds/288 candidates")
    rng=random.Random(seed)
    m=len(values)
    v=sorted(sum(values[rng.randrange(m)] for _ in range(m))/m for _ in range(n))
    return {
        "observed":sum(values)/m,
        "ci95_low":v[int(.025*n)],
        "ci95_high":v[int(.975*n)-1],
        "iterations":n,"seed":seed,
        "resampling_unit":"24_tasks_clusters_of_two_related_seeds",
    }


def verify_frozen(public:Path,private:Path,manifest:Path,gen_dir:Path)->tuple[list[dict],dict,dict]:
    m=json.loads(manifest.read_text(encoding="utf-8"))
    out_path=gen_dir/"generation_summary.json"
    frozen=json.loads(out_path.read_text(encoding="utf-8"))
    identity=json.loads((gen_dir/"identity.json").read_text(encoding="utf-8"))
    checkpoint=gen_dir/"generation_checkpoint.jsonl"
    if (m.get("experiment")!=EXPERIMENT
        or m.get("dataset_revision")!=DATA_REVISION
        or m["sha256"]["public_prompts"]!=PROMPT_SHA
        or m["sha256"]["private_augmented_oracle"]!=ORACLE_SHA
        or digest(public)!=PROMPT_SHA
        or frozen.get("status")!="MBPPPLUS_GENERATION_FROZEN"
        or frozen.get("answers_generated")!=TOTAL
        or frozen.get("source_script_sha256") is None
        or frozen.get("generation_checkpoint_sha256")!=digest(checkpoint)
        or frozen.get("oracle_labels_opened") is not False
        or frozen.get("heldout_oracle_never_opened") is not True
        or frozen.get("private_test_reference_code_never_opened") is not True
        or identity.get("expected_outputs")!=TOTAL
        or frozen.get("task_count")!=N_TASKS):
        raise ValueError("cannot read hidden MBPP+ oracle before 288 generations have been frozen")
    # First access to hidden oracle bytes must come AFTER checking that
    # all model generations and their SHA were frozen.
    if digest(private)!=ORACLE_SHA:
        raise ValueError("hidden augmented MBPP+ oracle source SHA changed")
    prompts=read_jsonl(public)
    if len(prompts)!=N_TASKS or len({r["task_id"] for r in prompts})!=N_TASKS:
        raise ValueError("frozen public benchmark cohort incomplete")
    generated=checkpoint_records(checkpoint,{r["task_id"] for r in prompts})
    expected={(row["task_id"],seed,j,arm) for row in prompts
              for seed in SEEDS for j in range(N_DRAWS) for arm in ARMS}
    if len(generated)!=TOTAL or set(generated)!=expected:
        raise ValueError("incomplete 24x2x2x3 model generation matrix")
    return prompts,generated,frozen


def verdict_key(row:dict)->tuple:
    return (row["task_id"],row["seed"],row["sample_id"],row["arm"])


def produce_metrics(tasks:list[dict],rows:dict)->dict:
    metrics={arm:{
        "candidate_pass":0,"candidate_count":N_TASKS*len(SEEDS)*N_DRAWS,
        "solved_at1":0,"solved_at2":0,
        "syntax_valid":0,"errors":Counter(),
        "by_seed":{str(seed):{"correct_candidates":0,"solved_at1":0,"solved_at2":0} for seed in SEEDS},
    } for arm in ARMS}
    per_task=[]
    for task in tasks:
        qid=task["task_id"]
        values={"task_id_sha256":hashlib.sha256(qid.encode()).hexdigest()}
        for arm in ARMS:
            per_seed=[]
            for seed in SEEDS:
                pair=[rows[qid,seed,j,arm] for j in range(N_DRAWS)]
                successes=[bool(x["passed"]) for x in pair]
                item=metrics[arm]
                n_correct=sum(successes)
                item["candidate_pass"]+=n_correct
                item["solved_at1"]+=int(successes[0])
                item["solved_at2"]+=int(any(successes))
                item["by_seed"][str(seed)]["correct_candidates"]+=n_correct
                item["by_seed"][str(seed)]["solved_at1"]+=int(successes[0])
                item["by_seed"][str(seed)]["solved_at2"]+=int(any(successes))
                item["syntax_valid"]+=sum(x["python_syntax_valid"] for x in pair)
                item["errors"].update(x["verdict"] for x in pair if not x["passed"])
                per_seed.append({
                    "correct_fraction":n_correct/N_DRAWS,
                    "at1":int(successes[0]),
                    "at2":int(any(successes)),
                })
            values[arm]={
                "correct_fraction":sum(x["correct_fraction"] for x in per_seed)/len(SEEDS),
                "at1":sum(x["at1"] for x in per_seed)/len(SEEDS),
                "at2":sum(x["at2"] for x in per_seed)/len(SEEDS),
            }
        per_task.append(values)
    for arm in ARMS:
        m=metrics[arm]
        m["candidate_accuracy"]=m["candidate_pass"]/m["candidate_count"]
        m["at1_rate"]=m["solved_at1"]/(N_TASKS*len(SEEDS))
        m["observed_solved_at2_rate"]=m["solved_at2"]/(N_TASKS*len(SEEDS))
        m["errors"]=dict(sorted(m["errors"].items()))
    paired={}
    for left,right in (("sft","new_grpo"),("sft","previous_grpo"),
                       ("previous_grpo","new_grpo")):
        paired[f"{right}_minus_{left}"]={
            metric:ci([p[right][metric]-p[left][metric] for p in per_task])
            for metric in ("correct_fraction","at1","at2")
        }
    return {
        "metric_scope":"MBPP+ hidden augmented assertions in a pinned Docker image, not native EvalPlus CLI original leaderboard",
        "num_tasks":N_TASKS,
        "num_task_seed_pairs":N_TASKS*len(SEEDS),
        "total_candidates":TOTAL,
        "arms":metrics,"paired_task_bootstrap":paired,
        "interpretation":"Observed solved@2 is at least one correct among two sampled programs, NOT an unbiased pass@2 estimator.",
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--public",required=True,type=Path)
    parser.add_argument("--private-oracle",required=True,type=Path)
    parser.add_argument("--manifest",required=True,type=Path)
    parser.add_argument("--generation-dir",required=True,type=Path)
    parser.add_argument("--workers",type=int,default=3)
    args=parser.parse_args()
    if not 1<=args.workers<=4:
        raise ValueError("bounded safe Docker scoring workers 1..4 required")
    tasks,generated,summary=verify_frozen(
        args.public,args.private_oracle,args.manifest,args.generation_dir
    )
    # Only NOW read hidden augmented test oracle.
    hidden=read_jsonl(args.private_oracle)
    if len(hidden)!=N_TASKS or [r["task_id"] for r in hidden]!=[r["task_id"] for r in tasks]:
        raise ValueError("MBPP+ hidden oracle task IDs/selection order changed")
    refs={r["task_id"]:r for r in hidden}
    image_sha()
    path=args.generation_dir/"scored_checkpoint.jsonl"
    scored={}
    if path.exists():
        for row in read_jsonl(path):
            key=verdict_key(row)
            if (key in scored or key not in generated
                or row["answer_sha256"]!=generated[key]["answer_sha256"]
                or type(row.get("passed")) is not bool):
                raise ValueError("scoring resume checkpoint mutated/duplicated")
            scored[key]=row
    pending=[key for key in sorted(generated) if key not in scored]
    print(f"EXP019 scorer pinned private oracle, existing={len(scored)}, pending={len(pending)}",flush=True)

    def evaluate(key):
        sample=generated[key]
        code,extraction=normalize_code(sample["answer"])
        try:
            ast.parse(code)
            valid=bool(code.strip())
        except (SyntaxError,MemoryError,RecursionError,ValueError):
            valid=False
        if valid:
            judged=test_program(code,refs[key[0]])
        else:
            judged={"passed":False,"error":"python_syntax_invalid"}
        verdict={
            "task_id":key[0],"seed":key[1],"sample_id":key[2],"arm":key[3],
            "answer_sha256":sample["answer_sha256"],
            "normalized_code_sha256":hashlib.sha256(code.encode()).hexdigest(),
            "normalizer":extraction,"python_syntax_valid":valid,
            "passed":judged["passed"],"verdict":judged["error"],
        }
        return key,verdict
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future={pool.submit(evaluate,key):key for key in pending}
        for f in as_completed(future):
            key,row=f.result()
            if key in scored:
                raise ValueError("duplicate concurrently returned score")
            with path.open("a",encoding="utf-8") as handle:
                handle.write(json.dumps(row,ensure_ascii=False)+"\n")
                handle.flush()
            scored[key]=row
            if len(scored)%24==0:
                print(f"EXP019 MBPP+ augmented Docker {len(scored)}/{TOTAL}",flush=True)
    if len(scored)!=TOTAL or set(scored)!=set(generated):
        raise ValueError("all 288 real candidates were not scored")
    result={
        "experiment":"EXP-019-THREE-7B-MBPPPLUS-EXTERNAL",
        "status":"completed_all288_heldout_Docker_scores",
        "benchmark_dataset_revision":DATA_REVISION,
        "source_cohort_manifest_sha256":digest(args.manifest),
        "public_task_sha256":digest(args.public),
        "private_augmented_oracle_sha256":digest(args.private_oracle),
        "generation_checkpoint_sha256":digest(args.generation_dir/"generation_checkpoint.jsonl"),
        "Docker_score_checkpoint_sha256":digest(path),
        "model_arms_sha256":summary["arms_sha256"],
        "Docker_image_sha256":IMAGE_ID,
        "generation_hidden_test_accessed":False,
        "all_generated_codes_executed_only_in_Docker":True,
        **produce_metrics(tasks,scored),
        "limitations":"Historical public MBPP+ with fixed 30sec Docker judging and public base test examples, not official original leaderboard nor time-forward blind test. Task count24, seeds repeated within a task.",
    }
    out=args.generation_dir/"score_summary.json"
    if out.exists():
        if json.loads(out.read_text())!=result:
            raise FileExistsError("refusing overwrite of different fixed benchmark report")
    else:
        out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
