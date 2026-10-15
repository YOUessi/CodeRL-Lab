"""EXP-018: read-only 7B GRPO 576-candidate failure and coverage audit.

Never executes the model, test labels or generated Python code. Consumes only
the already frozen generation checkpoint, Docker verdict checkpoint and
summary. Independently reconstructs 3x192 counts and verifies provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from coderl_lab.analysis.train017_7b_generate import (
    ARMS,SEEDS,N_TASKS,N_DRAWS,TOTAL,checkpoint_records,read_jsonl,
)
from coderl_lab.analysis.train017_7b_docker_score import group_scores
from coderl_lab.datasets.train017_new_algorithms import digest

EXPECTED_GENERATION_SHA="8960a134d2cdfcd850f93ce7106624906e337f8a53feff01f35d36dbb79efd92"
EXPECTED_DOCKER_SHA="57fe0efc279f856d46273defe0acc24bdd899de91441a5ba78720244fa356f7d"
EXPECTED_GENERATION_SUMMARY_SHA="8e21f47eef7662e328c968e0ebb81e599950c8ba705ccdf537d8884a4eeeabab"
EXPECTED_DOCKER_SUMMARY_SHA="3c0e0d22bbf093fbdf01e82f182f2149aa13bddc48aba64ea0621b2483234d23"


def aggregate(frozen:dict,verdicts:dict)->dict:
    if set(frozen)!=set(verdicts) or len(frozen)!=TOTAL:
        raise ValueError("all 576 aligned predictions and scores required")
    stats={a:{"by_family":{}, "by_seed":{}, "by_sample":{},
              "error_histogram":dict(Counter(v["verdict"] for k,v in verdicts.items() if k[3]==a)),
              "generated_source_length":{
                  "mean_chars":sum(len(v["answer"]) for k,v in frozen.items() if k[3]==a)/192,
                  "max_chars":max(len(v["answer"]) for k,v in frozen.items() if k[3]==a),
                  "nonempty":sum(bool(v["answer"].strip()) for k,v in frozen.items() if k[3]==a),
                  "hit_256_token_cap":sum(v["generated_tokens"]>=256 for k,v in frozen.items() if k[3]==a),
              }} for a in ARMS}
    ids=sorted({key[0] for key in frozen})
    if len(ids)!=N_TASKS:
        raise ValueError("must contain 24 distinct heldout tasks")
    families=sorted({qid.split("/")[1] for qid in ids})
    for arm in ARMS:
        for family in families:
            subset=[q for q in ids if q.split("/")[1]==family]
            values=[
                verdicts[q,s,n,arm]
                for q in subset for s in SEEDS for n in range(N_DRAWS)
            ]
            stats[arm]["by_family"][family]={
                "task_count":len(subset),
                "candidate_count":len(values),
                "correct":sum(v["all_passed"] for v in values),
                "mean_reward":sum(v["reward"] for v in values)/len(values),
                "syntax_invalid":sum(not v["python_valid"] for v in values),
                "entry_invalid":sum(not v["entry_valid"] for v in values),
                "timeouts":sum(v["verdict"]=="candidate_timeout" for v in values),
                "solved4_task_seed_pairs":sum(
                    any(verdicts[q,s,n,arm]["all_passed"] for n in range(N_DRAWS))
                    for q in subset for s in SEEDS
                ),
            }
        for seed in SEEDS:
            batch=[verdicts[q,seed,n,arm] for q in ids for n in range(N_DRAWS)]
            stats[arm]["by_seed"][str(seed)]={
                "correct":sum(v["all_passed"] for v in batch),
                "total":len(batch),
                "solved4":sum(
                    any(verdicts[q,seed,n,arm]["all_passed"] for n in range(N_DRAWS))
                    for q in ids
                ),
                "sample0_correct":sum(verdicts[q,seed,0,arm]["all_passed"] for q in ids),
            }
        for draw in range(N_DRAWS):
            batch=[verdicts[q,s,draw,arm] for q in ids for s in SEEDS]
            stats[arm]["by_sample"][str(draw)]={
                "correct":sum(v["all_passed"] for v in batch),
                "total":len(batch),
            }
    task_transitions={}
    for baseline,candidate in (("sft","new_grpo"),("sft","previous_grpo"),("previous_grpo","new_grpo")):
        flips=Counter()
        per_task_delta=Counter()
        all_winners_losers=[]
        for q in ids:
            gains=[];losses=[]
            for s in SEEDS:
                old=[verdicts[q,s,n,baseline]["all_passed"] for n in range(N_DRAWS)]
                new=[verdicts[q,s,n,candidate]["all_passed"] for n in range(N_DRAWS)]
                before=any(old);after=any(new)
                flips[f"{int(before)}_to_{int(after)}"]+=1
                gains.append(sum(new)-sum(old))
                losses.append(sum(new)-sum(old))
            delta=sum(gains)
            per_task_delta["positive" if delta>0 else "negative" if delta<0 else "zero"]+=1
            all_winners_losers.append(delta)
        task_transitions[f"{candidate}_minus_{baseline}"]={
            "task_seed_pair_solved4_transitions":dict(sorted(flips.items())),
            "task_level_candidate_correct_delta_signs":dict(per_task_delta),
            "task_level_candidate_correct_delta_values":dict(sorted(Counter(all_winners_losers).items())),
            "extra_correct_candidates":sum(all_winners_losers),
            "positive_over_zero_net_tasks":sum(x>0 for x in all_winners_losers),
            "negative_over_zero_net_tasks":sum(x<0 for x in all_winners_losers),
        }
    exact_code_cross_arm={}
    for first,second in (("sft","new_grpo"),("sft","previous_grpo"),("previous_grpo","new_grpo")):
        exact_code_cross_arm[f"{first}_vs_{second}"]={
            "same_exact_generated_text_at_same_task_seed_draw":sum(
                frozen[q,s,n,first]["answer_sha256"]==frozen[q,s,n,second]["answer_sha256"]
                for q in ids for s in SEEDS for n in range(N_DRAWS)
            ),
            "pairs_compared":N_TASKS*len(SEEDS)*N_DRAWS,
        }
    return {
        "experiment":"EXP-018-TRAIN017-7B-ERROR-ATTRIBUTION",
        "status":"completed_frozen_576_readonly",
        "tasks":N_TASKS,"seed_task_pairs":N_TASKS*len(SEEDS),
        "samples_per_arm":192,"all_candidates":TOTAL,
        "by_arm":stats,
        "coverage_transitions":task_transitions,
        "exact_generated_text_overlaps":exact_code_cross_arm,
        "interpretation_boundaries":[
            "Read-only retrospective after scoring, no new independent benchmark.",
            "Time-outs are candidate-level execution outcomes and are not proof of malicious code.",
            "Exact text similarity is not algorithm semantic equivalence.",
            "Different TRAIN016/017 training data/budgets prevent a pure training-step causal conclusion.",
        ],
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--public",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    original_generation=a.source/"generation_checkpoint.jsonl"
    original_docker=a.source/"docker_scored_checkpoint.jsonl"
    generation_summary=a.source/"generation_summary.json"
    docker_summary=a.source/"docker_score_summary.json"
    for path,expected in (
        (original_generation,EXPECTED_GENERATION_SHA),
        (original_docker,EXPECTED_DOCKER_SHA),
        (generation_summary,EXPECTED_GENERATION_SUMMARY_SHA),
        (docker_summary,EXPECTED_DOCKER_SUMMARY_SHA),
    ):
        if digest(path)!=expected:
            raise ValueError(f"frozen TRAIN017 checkpoint was replaced: {path}")
    docs=read_jsonl(a.public)
    if len(docs)!=24 or len({r["task_id"] for r in docs})!=24:
        raise ValueError("not 24 frozen public questions")
    generated=checkpoint_records(original_generation,{r["task_id"] for r in docs})
    scored={}
    for r in read_jsonl(original_docker):
        key=(r["task_id"],r["seed"],r["sample_id"],r["arm"])
        if key in scored:
            raise ValueError("duplicate scored answer key")
        if key not in generated or r["answer_sha256"]!=generated[key]["answer_sha256"]:
            raise ValueError("not the same generated model answer")
        scored[key]=r
    recorded=json.loads(docker_summary.read_text())
    independently=group_scores(docs,scored,generated)
    for k in ("status","task_count","model_arms","total_candidates","seeds","metrics","pairwise_task_cluster_bootstrap"):
        if independently[k]!=recorded[k]:
            raise ValueError(f"original recorded metrics cannot be independently recovered: {k}")
    result=aggregate(generated,scored)
    result["source_sha256"]={
        "generation_checkpoint":EXPECTED_GENERATION_SHA,
        "docker_verdict_checkpoint":EXPECTED_DOCKER_SHA,
        "generation_summary":EXPECTED_GENERATION_SUMMARY_SHA,
        "scoring_summary":EXPECTED_DOCKER_SUMMARY_SHA,
        "analysis_source":digest(Path(__file__)),
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    if a.out.exists():
        if json.loads(a.out.read_text())!=result:
            raise FileExistsError("refusing overwrite of EXP018 analysis")
    else:
        a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
