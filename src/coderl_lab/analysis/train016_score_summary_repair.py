"""TRAIN-016 immutable 96-verdict score aggregator and SFT pass@4 label correction.

The original scorer accidentally checked arm == base instead of arm == sft
while summing SFT solved_at4. This only recomputes aggregate metrics from
previously frozen Docker labels; it never reruns models or Docker.
"""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def paired_ci(values):
    if len(values)!=12:raise ValueError("require 12 paired tasks")
    rng=random.Random(42);n=len(values)
    v=sorted(sum(values[rng.randrange(n)] for j in range(n))/n for i in range(20000))
    return {"estimate":sum(values)/n,"ci95_low":v[500],
            "ci95_high":v[19499],"iterations":20000,"seed":42}

def recompute(items):
    if len(items)!=96:raise ValueError("require 96 Docker verdict rows")
    lookup=defaultdict(lambda:{"sft":{},"grpo":{}})
    for row in items:
        q,arm,n=row["task_id"],row["arm"],int(row["sample_id"])
        if arm not in ("sft","grpo") or n not in range(4) or n in lookup[q][arm]:
            raise ValueError("bad/duplicate model sample key")
        if type(row.get("all_passed")) is not bool:
            raise ValueError("invalid sandbox correctness verdict")
        lookup[q][arm][n]=row
    if len(lookup)!=12:raise ValueError("require 12 task IDs")
    per_task=[]
    for q,group in sorted(lookup.items()):
        one={"task_id":q}
        for arm in ("sft","grpo"):
            if set(group[arm])!=set(range(4)):
                raise ValueError("each arm must have 4 samples")
            batch=[group[arm][i] for i in range(4)]
            one[arm+"_at1"]=int(batch[0]["all_passed"])
            one[arm+"_at4"]=int(any(item["all_passed"] for item in batch))
            one[arm+"_correct"]=sum(item["all_passed"] for item in batch)
            one[arm+"_reward"]=sum(float(item["partial_reward"]) for item in batch)
        per_task.append(one)
    return {
        "tasks":12,"candidates":96,
        "scores_by_arm":{
            arm:{
                "candidate_correct":sum(t[arm+"_correct"] for t in per_task),
                "solved_at1":sum(t[arm+"_at1"] for t in per_task),
                "solved_at4":sum(t[arm+"_at4"] for t in per_task),
                "partial_reward_mean":sum(t[arm+"_reward"] for t in per_task)/48,
            } for arm in ("sft","grpo")
        },
        "paired_task_bootstrap":{
            "pass_at1_diff":paired_ci([t["grpo_at1"]-t["sft_at1"] for t in per_task]),
            "pass_at4_diff":paired_ci([t["grpo_at4"]-t["sft_at4"] for t in per_task]),
            "candidate_correct_fraction_diff":paired_ci([(t["grpo_correct"]-t["sft_correct"])/4 for t in per_task]),
        },
    }

def main():
    p=argparse.ArgumentParser()
    for name in ("scored","original","generations","output"):
        p.add_argument("--"+name,required=True,type=Path)
    a=p.parse_args()
    old=json.loads(a.original.read_text())
    if (old.get("status")!="completed_heldout_baseline_grpo"
        or old.get("model_arms")!=["sft","grpo"]
        or old.get("baseline_is_7b_sft_not_unadapted_backbone") is not True
        or old.get("score_checkpoint_sha256")!=digest(a.scored)
        or old.get("generated_checkpoint_sha256")!=digest(a.generations)):
        raise ValueError("original frozen report / checkpoints mismatch")
    rows=[json.loads(line) for line in a.scored.read_text().splitlines() if line.strip()]
    changed=recompute(rows)
    for arm in ("sft","grpo"):
        for metric in ("candidate_correct","solved_at1","partial_reward_mean"):
            if abs(changed["scores_by_arm"][arm][metric]-old["scores_by_arm"][arm][metric])>1e-10:
                raise ValueError("unrelated previously correct metric changed")
    if changed["scores_by_arm"]["grpo"]["solved_at4"]!=old["scores_by_arm"]["grpo"]["solved_at4"]:
        raise ValueError("GRPO summary unexpectedly changed")
    result={
        "experiment":"TRAIN-016-7B-SFT-VS-GRPO-HOLDOUT",
        "status":"corrected_report_v2_from_frozen_original",
        "repair":"Original code used arm == base rather than arm == sft for SFT solved_at4.",
        "raw_original_score_summary_sha256":digest(a.original),
        "raw_96_Docker_verdicts_sha256":digest(a.scored),
        "raw_96_generations_sha256":digest(a.generations),
        "model_arms":["sft","grpo"],
        "regenerated_codes":False,"executed_new_Docker_tests":False,
        "historical_small_synthetic_exploratory_benchmark":True,
        **changed,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():
        if json.loads(a.output.read_text())!=result:
            raise FileExistsError("refusing different corrected report overwrite")
    else:a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
