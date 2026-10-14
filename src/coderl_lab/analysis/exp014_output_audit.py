"""EXP-014 label-free diagnostics for 64x3 generated HH model answers.

Does not open HH chosen/rejected preference labels or conduct quality ratings.
Reports only length, stop/cap and exact output similarity; aggregate only.
"""
from __future__ import annotations
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from coderl_lab.datasets.exp014_hh_helpful import hash_file
from coderl_lab.analysis.exp014_blind_generate import ARMS, PUBLIC64_SHA, output_sha


def summarize(rows:list[dict])->dict:
    by_arm=defaultdict(list)
    by_task=defaultdict(dict)
    for r in rows:
        if r["arm"] not in ARMS or output_sha(r["answer"])!=r["answer_sha256"]:
            raise ValueError("unrecognized/mutated model completion")
        by_arm[r["arm"]].append(r)
        if r["arm"] in by_task[r["task_id"]]:
            raise ValueError("duplicate arm response")
        by_task[r["task_id"]][r["arm"]]=r
    if len(by_task)!=64 or any(set(group)!=set(ARMS) for group in by_task.values()):
        raise ValueError("64 tasks x3 genuine completions required")
    stat={}
    for arm in ARMS:
        group=by_arm[arm]
        chars=[len(x["answer"]) for x in group]
        tokens=[int(x["generated_tokens"]) for x in group]
        stat[arm]={
            "responses":len(group),
            "nonempty_answers":sum(bool(x["answer"].strip()) for x in group),
            "mean_chars":statistics.mean(chars),
            "median_chars":statistics.median(chars),
            "mean_generated_tokens":statistics.mean(tokens),
            "hit_224_generated_token_limit":sum(n>=224 for n in tokens),
        }
    pairwise={}
    for i,a in enumerate(ARMS):
        for b in ARMS[i+1:]:
            pairwise[f"{a}_vs_{b}"]=sum(
                x[a]["answer_sha256"]==x[b]["answer_sha256"]
                for x in by_task.values()
            )
    return {
        "experiment":"EXP-014-GENERATION-STRUCTURAL-AUDIT",
        "status":"label_free_completed",
        "tasks":64,"candidate_answers":192,
        "by_model":stat,
        "same_exact_answer_task_counts":pairwise,
        "quality_judgements_produced":0,
        "reviewers_produced":0,
        "interpretation":"Response length, stop/cap rate and exact text equality are NOT measures of helpfulness.",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--generation-dir",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    model_summary=json.loads((args.generation_dir/"summary.json").read_text())
    manifest=json.loads(args.manifest.read_text())
    checkpoint=args.generation_dir/"generation_checkpoint.jsonl"
    if (model_summary.get("status")!="generated_blinded_needs_human_review"
        or model_summary.get("answers_generated")!=192
        or hash_file(checkpoint)!=model_summary.get("generation_checkpoint_sha256")
        or manifest["sha256"]["generation_prompts"]!=PUBLIC64_SHA):
        raise ValueError("model generation not fully frozen")
    rows=[json.loads(s) for s in checkpoint.read_text(encoding="utf-8").splitlines() if s.strip()]
    result={
        **summarize(rows),
        "checkpoint_sha256":hash_file(checkpoint),
        "source_generation_sha256":hash_file(args.generation_dir/"identity.json"),
        "blinded_packets_sha256":model_summary["review_packet_sha256"],
        "human_ratings_status":"NOT_STARTED",
        "raw_candidate_responses_exported":False,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():
        if json.loads(args.output.read_text())!=result:
            raise FileExistsError("generation audit output changed")
    else:
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
