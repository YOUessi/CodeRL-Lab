"""EXP-014 posthoc metric sensitivity: response length and exact paired tests.

Read-only aggregate of 256 frozen HH helpful-base reference pairs. No new
model generation or adaptation. Contrasts raw sequence logP versus mean
per response token, with exact McNemar binomial p for sparse discordance.
The test dataset is historical public HH and this diagnostic is posthoc:
do NOT treat nominal p as independent prospectively calibrated claims.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from coderl_lab.datasets.exp014_hh_helpful import hash_file

ARMS=("sft","bf16_dpo","fp32_dpo")
METHODS=("sum","mean")


def exact_mcnemar(gain:int,loss:int)->float:
    n=gain+loss
    if n==0:return 1.
    tail=sum(math.comb(n,k) for k in range(min(gain,loss)+1))/(2**n)
    return min(1.,2*tail)


def group_accuracy(rows:list[dict],arm:str,method:str)->dict:
    correct=[
        r for r in rows
        if (r[f"{arm}_chosen_{method}_logp"]
            > r[f"{arm}_rejected_{method}_logp"])
    ]
    return {
        "n":len(rows),
        "chosen_preferred":len(correct),
        "preference_accuracy":len(correct)/len(rows),
    }


def analyze(rows:list[dict])->dict:
    if len(rows)!=256 or any(r.get("scorable") is not True for r in rows):
        raise ValueError("require all256 model-scored HH references for this exact diagnostic")
    by_length={}
    def length_group(row):
        d=int(row["chosen_target_tokens"])-int(row["rejected_target_tokens"])
        return "chosen_longer" if d>0 else ("same_token_length" if d==0 else "chosen_shorter")
    for kind in ("chosen_longer","chosen_shorter","same_token_length"):
        subset=[r for r in rows if length_group(r)==kind]
        by_length[kind]={
            "n":len(subset),
            "accuracy":{
                arm:{
                    typ:group_accuracy(subset,arm,typ) if subset else None
                    for typ in METHODS
                } for arm in ARMS
            },
        }
    comparisons={}
    for old,new in (("sft","bf16_dpo"),("sft","fp32_dpo"),("bf16_dpo","fp32_dpo")):
        result={}
        for typ in METHODS:
            a=[r[f"{old}_chosen_{typ}_logp"]>r[f"{old}_rejected_{typ}_logp"] for r in rows]
            b=[r[f"{new}_chosen_{typ}_logp"]>r[f"{new}_rejected_{typ}_logp"] for r in rows]
            gained=sum((not x) and y for x,y in zip(a,b))
            lost=sum(x and (not y) for x,y in zip(a,b))
            result[typ]={
                "old_correct":sum(a),
                "new_correct":sum(b),
                "new_minus_old_count":sum(b)-sum(a),
                "old_wrong_new_correct":gained,
                "old_correct_new_wrong":lost,
                "exact_two_sided_mcnemar_p":exact_mcnemar(gained,lost),
            }
        comparisons[f"{new}_minus_{old}"]=result
    return {
        "experiment":"EXP-014-LENGTH-METRIC-SENSITIVITY-AUDIT",
        "status":"posthoc_cpu_audit_completed",
        "n":len(rows),
        "chosen_mean_tokens":statistics.mean(r["chosen_target_tokens"] for r in rows),
        "rejected_mean_tokens":statistics.mean(r["rejected_target_tokens"] for r in rows),
        "chosen_minus_rejected_mean_tokens":statistics.mean(
            r["chosen_target_tokens"]-r["rejected_target_tokens"] for r in rows
        ),
        "length_strata":by_length,
        "exact_discordant_pair_tests":comparisons,
        "warning":"Posthoc metric sensitivity, historical public dataset. Exact paired p is descriptive and not adjusted for multiple comparisons; no human quality scores of model-generated responses are present.",
        "generated_response_quality_reviewed":False,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--preference-dir",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    summary=json.loads((args.preference_dir/"summary.json").read_text())
    checkpoint=args.preference_dir/"checkpoint.jsonl"
    if (summary.get("status")!="completed"
        or summary["metrics"]["pairs_total"]!=256
        or summary["metrics"]["pairs_scorable"]!=256
        or hash_file(checkpoint)!=summary["checkpoint_sha256"]):
        raise ValueError("cannot audit incomplete/mutated HH score checkpoint")
    rows=[json.loads(x) for x in checkpoint.read_text().splitlines() if x.strip()]
    if len(rows)!=256 or {r["index"] for r in rows}!=set(range(256)):
        raise ValueError("not exactly256 unique HH references")
    outcome=analyze(rows)
    outcome["source_checkpoint_sha256"]=hash_file(checkpoint)
    outcome["source_pref_summary_sha256"]=hash_file(args.preference_dir/"summary.json")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.out.exists():
        if json.loads(args.out.read_text())!=outcome:
            raise FileExistsError("refuse to overwrite different posthoc audit")
    else:
        args.out.write_text(json.dumps(outcome,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(outcome,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
