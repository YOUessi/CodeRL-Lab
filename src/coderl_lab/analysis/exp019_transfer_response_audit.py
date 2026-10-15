"""EXP-019 posthoc exact-code and correctness stability across MBPP+ three 7B arms.

Reads only previously frozen 288 model responses and 288 immutable Docker
verdicts. Never opens private oracle, never executes candidate code and
never generates model text. This is a retrospective analysis.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

ARMS=("sft","previous_grpo","new_grpo")
SEEDS=(42,314159)
N_DRAWS=2
TOTAL=24*len(SEEDS)*N_DRAWS*len(ARMS)
GEN_SHA="18344ffc824c46c07be0fce5f31350492e40f39baa87a7bee3a79ee607d48f58"
SCORE_SHA="b563cb50236ee5076a1d2a744779c2965696820ef72d968af489b51fb37bf99a"
BASE_SUMMARY_SHA="4e19781a0de74db9d76d8a2bd9277859f159258e4ce1db06c2a48956bfed66ee"

def digest(p:Path)->str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(path:Path)->list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def paired(gens:list[dict],judges:list[dict])->dict:
    if len(gens)!=TOTAL or len(judges)!=TOTAL:
        raise ValueError("must analyze the entire 288 output/scoring pair matrix")
    keys=set()
    gen={}
    score={}
    for row in gens:
        key=(row["task_id"],row["seed"],row["sample_id"],row["arm"])
        if (key in gen or key[1] not in SEEDS or key[2] not in range(N_DRAWS)
            or key[3] not in ARMS or digest_text(row["answer"])!=row["answer_sha256"]):
            raise ValueError("duplicate/invalid/changed output checkpoint")
        gen[key]=row
    for row in judges:
        key=(row["task_id"],row["seed"],row["sample_id"],row["arm"])
        if (key in score or key not in gen
            or row["answer_sha256"]!=gen[key]["answer_sha256"]
            or type(row.get("passed")) is not bool):
            raise ValueError("duplicate or mutated Docker verdict")
        score[key]=row
    if set(gen)!=set(score) or len({k[0] for k in gen})!=24:
        raise ValueError("incomplete identical frozen task ID matrix")
    task_ids={k[0] for k in gen}
    expected={(t,seed,j,arm) for t in task_ids for seed in SEEDS
              for j in range(N_DRAWS) for arm in ARMS}
    if expected!=set(gen):
        raise ValueError("not all task/seed/draw/arm combinations present")

    pairwise={}
    for a,b in (("sft","new_grpo"),("sft","previous_grpo"),
                ("previous_grpo","new_grpo")):
        raw_match=norm_match=0
        flip=Counter()
        by_seed={str(seed):{"raw_match":0,"success_flip":0} for seed in SEEDS}
        for qid in task_ids:
            for seed in SEEDS:
                for sample in range(N_DRAWS):
                    ka=(qid,seed,sample,a);kb=(qid,seed,sample,b)
                    x=gen[ka];y=gen[kb];r1=score[ka];r2=score[kb]
                    match=x["answer_sha256"]==y["answer_sha256"]
                    raw_match+=int(match)
                    norm_match+=int(r1["normalized_code_sha256"]==r2["normalized_code_sha256"])
                    flip[f"old{int(r1['passed'])}_new{int(r2['passed'])}"]+=1
                    by_seed[str(seed)]["raw_match"]+=int(match)
                    by_seed[str(seed)]["success_flip"]+=int(r1["passed"]!=r2["passed"])
        pairwise[f"{b}_minus_{a}"]={
            "paired_candidate_count":96,
            "raw_generation_exact_match":raw_match,
            "normalized_code_exact_match":norm_match,
            "different_raw_answers_with_same_result":96-raw_match-
                 (flip["old0_new1"]+flip["old1_new0"]),
            "status_flips":dict(sorted(flip.items())),
            "success_gains":flip["old0_new1"],
            "success_losses":flip["old1_new0"],
            "per_seed":by_seed,
        }
    m={}
    for arm in ARMS:
        rows=[r for k,r in gen.items() if k[3]==arm]
        tests=[r for k,r in score.items() if k[3]==arm]
        lengths=[int(r["generated_tokens"]) for r in rows]
        m[arm]={
            "mean_tokens":statistics.mean(lengths),
            "median_tokens":statistics.median(lengths),
            "hit256_token_limit":sum(x>=256 for x in lengths),
            "correct":sum(r["passed"] for r in tests),
            "syntax_valid":sum(r["python_syntax_valid"] for r in tests),
            "errors":dict(sorted(Counter(r["verdict"] for r in tests if not r["passed"]).items())),
        }
    return {
        "experiment":"EXP-019-MBPPPLUS-AFTER-FROZEN-SCORING-TRANSFER-DIAGNOSTIC",
        "status":"completed_no_rescore",
        "task_count":24,"samples_per_model":96,"paired_matrix_rows":TOTAL,
        "model_arms":list(ARMS),
        "by_model":m,
        "pairwise":pairwise,
        "raw_generated_programs_exported":False,
        "has_new_GPU_generation":False,
        "has_new_Docker_execution":False,
        "interpretive_boundary":"An unchanged success/failure status in 96 matching MBPP+ completions does not mean the 7B policies are exactly identical. Historical public benchmark and limited 24 tasks.",
    }

def digest_text(s:str)->str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--generation-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    source=a.generation_dir
    gp=source/"generation_checkpoint.jsonl"
    sp=source/"scored_checkpoint.jsonl"
    ss=source/"score_summary.json"
    if (digest(gp)!=GEN_SHA or digest(sp)!=SCORE_SHA
        or digest(ss)!=BASE_SUMMARY_SHA):
        raise ValueError("source 288 MBPP+ generation/scoring changed")
    baseline=json.loads(ss.read_text(encoding="utf-8"))
    if baseline["status"]!="completed_all288_heldout_Docker_scores":
        raise ValueError("not a completed real Docker benchmark")
    answer=paired(read(gp),read(sp))
    for arm in ARMS:
        if answer["by_model"][arm]["correct"]!=baseline["arms"][arm]["candidate_pass"]:
            raise ValueError("model correctness no longer matches frozen original benchmark")
    answer["sha256"]={"generation_checkpoint":GEN_SHA,
                      "scoring_checkpoint":SCORE_SHA,
                      "base_summary":BASE_SUMMARY_SHA,
                      "audit_source":digest(Path(__file__))}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    if a.out.exists():
        if json.loads(a.out.read_text())!=answer:
            raise FileExistsError("refusing changed retrospective audit")
    else:
        a.out.write_text(json.dumps(answer,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(answer,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":main()
