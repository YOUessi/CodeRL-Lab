"""EXP-019B: read-only exact paired 7B SFT vs GRPO external code audit.

Does not re-run models/Docker or read hidden MBPP+ oracles. Requires frozen
checkpoint hashes, full 24*4*2 output matrix and original score summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from coderl_lab.datasets.exp019b_mbppplus_freeze import digest
from coderl_lab.analysis.exp019b_7b_generate import read_jsonl,read_checkpoint,ARMS,DRAWS

def summarize(complete:dict,score:dict)->dict:
    if set(complete)!=set(score) or len(complete)!=192:
        raise ValueError("exact192 generated/scored candidate pairs required")
    tasks={q for q,i,a in complete}
    if len(tasks)!=24:
        raise ValueError("not exactly24 external benchmark tasks")
    counts=Counter()
    seen=Counter()
    failure_types={arm:Counter() for arm in ARMS}
    per_task={}
    for q in sorted(tasks):
        paired=[]
        for i in range(DRAWS):
            sft=complete[q,i,"sft"]
            grpo=complete[q,i,"grpo"]
            ss=score[q,i,"sft"]
            gs=score[q,i,"grpo"]
            for raw,graded in ((sft,ss),(grpo,gs)):
                if raw["code_sha256"]!=graded["code_sha256"]:
                    raise ValueError("scoring checkpoint doesn't match generated code")
            same=sft["code_sha256"]==grpo["code_sha256"]
            correct_same=ss["passed"]==gs["passed"]
            counts["pair_samples"]+=1
            counts["identical_generated_text"]+=int(same)
            counts["different_generated_text"]+=int(not same)
            counts["same_correctness"]+=int(correct_same)
            counts["correctness_flip"]+=int(not correct_same)
            if not same:
                counts["changed_text_both_correct"]+=int(ss["passed"] and gs["passed"])
                counts["changed_text_both_wrong"]+=int(not ss["passed"] and not gs["passed"])
                counts["changed_text_sft_only_correct"]+=int(ss["passed"] and not gs["passed"])
                counts["changed_text_grpo_only_correct"]+=int(gs["passed"] and not ss["passed"])
            for arm,item in (("sft",ss),("grpo",gs)):
                failure_types[arm][item["reason"]]+=1
            paired.append((same,ss["passed"],gs["passed"]))
        per_task[q]={
            "num_changed_text_draws":sum(not x[0] for x in paired),
            "num_correctness_flips":sum(x[1]!=x[2] for x in paired),
            "sft_correct":sum(x[1] for x in paired),
            "grpo_correct":sum(x[2] for x in paired),
        }
    task_class={
        "text_changed_tasks":sum(v["num_changed_text_draws"]>0 for v in per_task.values()),
        "all_four_text_identical_tasks":sum(v["num_changed_text_draws"]==0 for v in per_task.values()),
        "correctness_flip_tasks":sum(v["num_correctness_flips"]>0 for v in per_task.values()),
        "task_correct_count_increase":sum(v["grpo_correct"]>v["sft_correct"] for v in per_task.values()),
        "task_correct_count_decrease":sum(v["grpo_correct"]<v["sft_correct"] for v in per_task.values()),
    }
    return {
        "experiment":"EXP-019B-7B-MBPPPLUS-PAIRED-CODE-AUDIT",
        "status":"complete_readonly_192",
        "source_tasks":24,
        "matched_sample_pairs":96,
        "paired_sample_counts":dict(counts),
        "per_task_aggregate_no_raw_code":task_class,
        "by_arm_verdict_reasons":{a:dict(c) for a,c in failure_types.items()},
        "raw_program_source_exported":False,
        "read_hidden_MBPP_tests":False,
        "interpretation":"GRPO altered some code text but changed no pass/fail outcomes in fixed24-task subset. This is not proof two models are identical beyond the frozen sampled outputs.",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dir",type=Path,required=True)
    p.add_argument("--public",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    gen=a.dir/"checkpoint.jsonl"
    scored=a.dir/"docker_score_checkpoint.jsonl"
    generation=json.loads((a.dir/"generation_summary.json").read_text())
    evaluation=json.loads((a.dir/"score_summary.json").read_text())
    if (generation.get("status")!="all_samples_generated_frozen"
        or evaluation.get("status")!="completed_pinned_MBPPPlus_24_tasks"
        or generation["checkpoint_sha256"]!=digest(gen)
        or evaluation["candidate_scoring_checkpoint_sha256"]!=digest(scored)
        or evaluation["reference_solution_test_status"]["passed"]!=24):
        raise ValueError("MBPP+ gold environment/frozen generation/scores mismatch")
    public=read_jsonl(a.public)
    generated=read_checkpoint(gen,public)
    grades={}
    for row in read_jsonl(scored):
        key=(row["task_id"],row["sample_id"],row["arm"])
        if key in grades:
            raise ValueError("duplicate scoring key")
        grades[key]=row
    report=summarize(generated,grades)
    if (sum(report["by_arm_verdict_reasons"][arm].get("none",0) for arm in ARMS)!=100
        or report["paired_sample_counts"]["correctness_flip"]!=0):
        raise ValueError("expected fully frozen 50+50 and exact sample agreement from original score")
    report["source_sha256"]={
        "generation":digest(gen),
        "score":digest(scored),
        "generation_summary":digest(a.dir/"generation_summary.json"),
        "scoring_summary":digest(a.dir/"score_summary.json"),
        "source_code":digest(Path(__file__)),
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():
        if json.loads(a.output.read_text())!=report:
            raise FileExistsError("refusing to overwrite old external audit")
    else:a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
