"""EXP-018: immutable retrospective 7B GRPO reliability vs solved@4 audit.

Reads 576 already generated source answers and 576 Docker score verdicts.
It never executes candidate code, calls a model or reopens the test oracle.
The two random seeds are repeated observations WITHIN 24 task clusters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

MODEL_ARMS=("sft","previous_grpo","new_grpo")
SEEDS=(42,314159)
N_TASKS=24
SAMPLES_PER_TASK=4
EXPECTED_RECORDS=N_TASKS*len(SEEDS)*SAMPLES_PER_TASK*len(MODEL_ARMS)
GEN_SHA="8960a134d2cdfcd850f93ce7106624906e337f8a53feff01f35d36dbb79efd92"
SCORE_SHA="57fe0efc279f856d46273defe0acc24bdd899de91441a5ba78720244fa356f7d"
GEN_REPORT_SHA="8e21f47eef7662e328c968e0ebb81e599950c8ba705ccdf537d8884a4eeeabab"
DATA_SHAS={
    "train":"21a63fcef2e74d7e902480db9aabceb6b61d5f3bd4aff551e1959f7b8e38e4bb",
    "holdout_public":"3f8768827ba8cba10d3555a58a3de5e69ec197a0192f1adb40ea05ae13c0e7a8",
    "holdout_oracle":"2cea9653952f7357a5f0ea549a32a94d36e3a0b46a9332d5b7d46ee84978a027",
}


def digest(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path:Path)->list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def row_index(rows:list[dict], name:str)->dict:
    if len(rows)!=EXPECTED_RECORDS:
        raise ValueError(f"{name} must have all {EXPECTED_RECORDS} frozen rows")
    result={}
    for row in rows:
        key=(row["task_id"],row["seed"],row["sample_id"],row["arm"])
        if (key in result or key[1] not in SEEDS or key[2] not in range(SAMPLES_PER_TASK)
            or key[3] not in MODEL_ARMS):
            raise ValueError(f"duplicated/invalid {name} row key {key}")
        result[key]=row
    return result


def bootstrap(values:list[float],*,reps:int=20000,seed:int=42)->dict:
    if len(values)!=N_TASKS:
        raise ValueError("bootstrap resampling unit must be the 24 distinct tasks")
    rng=random.Random(seed)
    n=len(values)
    observations=sorted(
        sum(values[rng.randrange(n)] for _ in range(n))/n
        for _ in range(reps)
    )
    return {"observed":sum(values)/n,
            "ci95_low":observations[int(.025*reps)],
            "ci95_high":observations[int(.975*reps)-1],
            "iterations":reps,"random_seed":seed,
            "cluster":"one_task_with_two_correlated_seeds"}


def exact_sign_test(wins:int,losses:int)->float:
    from math import comb
    n=wins+losses
    if n==0:return 1.0
    p=2*sum(comb(n,i) for i in range(min(wins,losses)+1))/2**n
    return min(1.,p)


def validate_dataset(public:Path,manifest:Path,generation_summary:Path,score_summary:Path,
                     generations:Path,scored:Path)->list[str]:
    for file,expected in ((generations,GEN_SHA),(scored,SCORE_SHA),
                          (generation_summary,GEN_REPORT_SHA)):
        if digest(file)!=expected:
            raise ValueError("frozen generated/scored checkpoint or report SHA drift: "+str(file))
    frozen=json.loads(manifest.read_text(encoding="utf-8"))
    g=json.loads(generation_summary.read_text(encoding="utf-8"))
    s=json.loads(score_summary.read_text(encoding="utf-8"))
    if (frozen.get("experiment")!="TRAIN-017-7B-NEW-CODE-GRPO"
        or frozen.get("train_tasks")!=48
        or frozen.get("heldout_tasks")!=24
        or frozen.get("sha256")!=DATA_SHAS
        or digest(public)!=DATA_SHAS["holdout_public"]):
        raise ValueError("TRAIN017 frozen 48/24 source/manifest drift")
    if (g.get("status")!="all_576_generated_frozen"
        or g.get("answers_generated")!=EXPECTED_RECORDS
        or g.get("generation_checkpoint_sha256")!=GEN_SHA
        or s.get("status")!="completed_all576_docker_scored"
        or s.get("total_candidates")!=EXPECTED_RECORDS
        or s.get("frozen_outputs_sha256")!=GEN_SHA
        or s.get("docker_scored_checkpoint_sha256")!=SCORE_SHA
        or s.get("model_arms")!=list(MODEL_ARMS)
        or s.get("seeds")!=list(SEEDS)):
        raise ValueError("source TRAIN017 generation/Docker score summary not truly complete")
    public_rows=read_jsonl(public)
    ids=[r["task_id"] for r in public_rows]
    if len(ids)!=N_TASKS or len(set(ids))!=N_TASKS:
        raise ValueError("heldout prompt public file task count drift")
    if any(set(row)!={"task_id","task_kind","prompt"} for row in public_rows):
        raise ValueError("test answers leaked into source prompt view")
    return ids


def compute(generation_rows:list[dict],scored_rows:list[dict],
            task_ids:list[str],training_steps:int=32)->dict:
    generated=row_index(generation_rows,"generated answers")
    scores=row_index(scored_rows,"Docker test verdicts")
    if set(generated)!=set(scores):
        raise ValueError("model generation and Docker verdict keys differ")
    if set(task_ids)!={key[0] for key in generated} or len(task_ids)!=N_TASKS:
        raise ValueError("not all frozen 24 heldout questions represented")
    expected={(qid,seed,k,arm) for qid in task_ids for seed in SEEDS
              for k in range(SAMPLES_PER_TASK) for arm in MODEL_ARMS}
    if set(generated)!=expected:
        raise ValueError("expected full 24x2x4x3 freeze matrix")
    for key in expected:
        g=generated[key];s=scores[key]
        if (hashlib.sha256(g["answer"].encode("utf-8")).hexdigest()!=g["answer_sha256"]
            or g["answer_sha256"]!=s["answer_sha256"]
            or type(s["all_passed"]) is not bool
            or type(s["python_valid"]) is not bool
            or type(s["entry_valid"]) is not bool
            or s["verdict"] not in {"none","wrong_answer","candidate_timeout",
                                  "invalid_code","candidate_memory_limit"}
            or not 0.<=float(s["reward"])<=1.):
            raise ValueError(f"mutated/inconsistent model completion or Docker score {key}")
        if s["verdict"]=="none" and not s["all_passed"]:
            raise ValueError("success verdict not marked correct")
        if s["all_passed"] and s["verdict"]!="none":
            raise ValueError("full pass with failed Docker status")
    # Paired outcomes keyed by the same task, two fixed seeds and 4 draws.
    details=[]
    for qid in sorted(task_ids):
        for seed in SEEDS:
            entry={"task_id":qid,"family":qid.split("/")[1],
                   "algorithm_param":qid.rsplit("/",1)[-1],
                   "seed":seed,"arms":{}}
            for arm in MODEL_ARMS:
                a=[scores[qid,seed,j,arm] for j in range(SAMPLES_PER_TASK)]
                g=[generated[qid,seed,j,arm] for j in range(SAMPLES_PER_TASK)]
                ok=[bool(r["all_passed"]) for r in a]
                entry["arms"][arm]={
                    "correct_count":sum(ok),
                    "solved_at_k":[int(any(ok[:k])) for k in range(1,SAMPLES_PER_TASK+1)],
                    "mean_reward":sum(float(r["reward"]) for r in a)/SAMPLES_PER_TASK,
                    "unique_exact_answer_sha_count":len({r["answer_sha256"] for r in g}),
                    "unique_correct_exact_answer_sha_count":len(
                        {r["answer_sha256"] for r,s in zip(g,a) if s["all_passed"]}
                    ),
                    "token_cap_hits":sum(int(r["generated_tokens"]>=256) for r in g),
                    "mean_generated_tokens":sum(int(r["generated_tokens"]) for r in g)/SAMPLES_PER_TASK,
                }
            details.append(entry)
    if len(details)!=N_TASKS*len(SEEDS):
        raise ValueError("not exactly two repeated-seed details per task")

    arm_metrics={}
    families=sorted({q.split("/")[1] for q in task_ids})
    for arm in MODEL_ARMS:
        arr=[scores[key] for key in scores if key[3]==arm]
        arm_metrics[arm]={
            "candidate_correct":sum(r["all_passed"] for r in arr),
            "candidate_count":len(arr),
            "solved_at_k":[sum(d["arms"][arm]["solved_at_k"][k] for d in details)
                           for k in range(4)],
            "correct_by_seed":{str(seed):sum(
                scores[qid,seed,j,arm]["all_passed"]
                for qid in task_ids for j in range(4)
            ) for seed in SEEDS},
            "verdict_histogram":dict(Counter(r["verdict"] for r in arr)),
            "mean_reward":sum(float(r["reward"]) for r in arr)/len(arr),
            "mean_exact_distinct_sources_per_task_seed":sum(
                d["arms"][arm]["unique_exact_answer_sha_count"] for d in details)/len(details),
            "mean_exact_distinct_correct_sources_per_task_seed":sum(
                d["arms"][arm]["unique_correct_exact_answer_sha_count"] for d in details)/len(details),
            "max256_token_hits":sum(d["arms"][arm]["token_cap_hits"] for d in details),
        }
    family_metrics={}
    for fam in families:
        qs=[q for q in task_ids if q.split("/")[1]==fam]
        sub=[d for d in details if d["family"]==fam]
        family_metrics[fam]={
            "tasks":len(qs),
            "task_seed_pairs":len(sub),
            "models":{arm:{
                "correct_candidates":sum(scores[q,seed,j,arm]["all_passed"] for q in qs for seed in SEEDS for j in range(4)),
                "solved_at4":sum(d["arms"][arm]["solved_at_k"][3] for d in sub),
                "solved_at1":sum(d["arms"][arm]["solved_at_k"][0] for d in sub),
            } for arm in MODEL_ARMS},
        }
    comparison={}
    for left,right in (("sft","new_grpo"),("sft","previous_grpo"),
                       ("previous_grpo","new_grpo")):
        sample_flip=Counter()
        for q in task_ids:
            for seed in SEEDS:
                for j in range(4):
                    x=scores[q,seed,j,left]["all_passed"]
                    y=scores[q,seed,j,right]["all_passed"]
                    sample_flip[f"old{int(x)}_new{int(y)}"]+=1
        group_flip=Counter()
        deltas={
            "candidate_accuracy":[],
            "solved_at1":[],
            "solved_at2":[],
            "solved_at3":[],
            "solved_at4":[],
            "mean_partial_reward":[],
            "distinct_raw_answers":[],
            "distinct_correct_raw_answers":[],
        }
        # Cluster unit = one original task with two random seeds.
        for qid in sorted(task_ids):
            per_seed=[d for d in details if d["task_id"]==qid]
            for d in per_seed:
                l=d["arms"][left];r=d["arms"][right]
                group_flip[f"old{l['solved_at_k'][3]}_new{r['solved_at_k'][3]}"]+=1
            deltas["candidate_accuracy"].append(sum(
                (d["arms"][right]["correct_count"]-d["arms"][left]["correct_count"])/4
                for d in per_seed)/len(per_seed))
            for k in range(4):
                deltas[f"solved_at{k+1}"].append(sum(
                    d["arms"][right]["solved_at_k"][k]-d["arms"][left]["solved_at_k"][k]
                    for d in per_seed)/len(per_seed))
            deltas["mean_partial_reward"].append(sum(
                d["arms"][right]["mean_reward"]-d["arms"][left]["mean_reward"]
                for d in per_seed)/len(per_seed))
            for attr,metric in (("unique_exact_answer_sha_count","distinct_raw_answers"),
                                ("unique_correct_exact_answer_sha_count","distinct_correct_raw_answers")):
                deltas[metric].append(sum(
                    d["arms"][right][attr]-d["arms"][left][attr]
                    for d in per_seed)/len(per_seed))
        comparison[f"{right}_minus_{left}"]={
            "sample_correctness_transitions":dict(sorted(sample_flip.items())),
            "sample_correctness_net_gain":sample_flip["old0_new1"]-sample_flip["old1_new0"],
            "task_seed_solved4_transitions":dict(sorted(group_flip.items())),
            "task_seed_newly_covered":group_flip["old0_new1"],
            "task_seed_lost_coverage":group_flip["old1_new0"],
            "already_covered_reliability_gain":sum(
                d["arms"][right]["correct_count"]-d["arms"][left]["correct_count"]
                for d in details
                if d["arms"][left]["solved_at_k"][3] and d["arms"][right]["solved_at_k"][3]
            ),
            "task_cluster_bootstrap":{
                metric:bootstrap(v) for metric,v in deltas.items()
            },
        }
        if comparison[f"{right}_minus_{left}"]["sample_correctness_net_gain"] != (
            arm_metrics[right]["candidate_correct"]-arm_metrics[left]["candidate_correct"]
        ):
            raise ValueError("sample transitions do not sum to observed candidate gain")
        if (group_flip["old0_new1"]-group_flip["old1_new0"]) != (
            arm_metrics[right]["solved_at_k"][3]-arm_metrics[left]["solved_at_k"][3]
        ):
            raise ValueError("coverage transitions do not sum to observed solved@4 gain")

    return {
        "experiment":"EXP-018-7B-RELIABILITY-VS-COVERAGE",
        "status":"frozen_readonly_audit_completed",
        "qualifier":"RETROSPECTIVE, same previously inspected 24 synthetic algorithm tasks, 2 correlated seeds",
        "task_count":N_TASKS,
        "seed_count":len(SEEDS),
        "task_seed_pairs":len(details),
        "candidate_count":EXPECTED_RECORDS,
        "model_arms":list(MODEL_ARMS),
        "arm_metrics":arm_metrics,
        "family_metrics":family_metrics,
        "paired_comparison":comparison,
        "task_seed_summaries":[{
            "task_id":d["task_id"],"seed":d["seed"],
            "family":d["family"],"algorithm_param":d["algorithm_param"],
            "per_arm_correct_count":{arm:d["arms"][arm]["correct_count"] for arm in MODEL_ARMS},
            "per_arm_solved_at_k":{arm:d["arms"][arm]["solved_at_k"] for arm in MODEL_ARMS},
        } for d in details],
        "raw_generated_code_exported":False,
        "no_new_GPU_generation":True,
        "no_new_Docker_execution":True,
        "model_training_not_changed":True,
        "limitations":"Different exact code text is not a different algorithm; the sample-level flips of stochastic completions are descriptive. Bootstrap across only 24 families/parameter instances is exploratory and not a new prospective heldout, without multiple-testing correction.",
    }


def main():
    p=argparse.ArgumentParser()
    for field in ("manifest","public","generation-summary","docker-summary",
                  "generations","scores","out"):
        p.add_argument("--"+field,type=Path,required=True)
    a=p.parse_args()
    public_ids=validate_dataset(
        a.public,a.manifest,a.generation_summary,a.docker_summary,
        a.generations,a.scores
    )
    report=compute(read_jsonl(a.generations),read_jsonl(a.scores),public_ids)
    scored_summary=json.loads(a.docker_summary.read_text(encoding="utf-8"))
    for arm,metrics in report["arm_metrics"].items():
        expected=scored_summary["metrics"][arm]
        if (metrics["candidate_correct"]!=expected["candidate_correct"]
            or metrics["solved_at_k"][0]!=expected["sample0_correct"]
            or metrics["solved_at_k"][3]!=expected["solved4"]
            or abs(metrics["mean_reward"]-expected["mean_partial_reward"])>1e-9):
            raise ValueError("original formal summary differs from independent frozen recalculation")
    report["source_sha256"]={
        "generation_checkpoint":digest(a.generations),
        "Docker_verdict_checkpoint":digest(a.scores),
        "generation_summary":digest(a.generation_summary),
        "Docker_summary":digest(a.docker_summary),
        "dataset_manifest":digest(a.manifest),
        "audit_code":digest(Path(__file__)),
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    if a.out.exists():
        if json.loads(a.out.read_text(encoding="utf-8"))!=report:
            raise FileExistsError("same 576 frozen verdicts produced a changed audit report")
    else:
        a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
