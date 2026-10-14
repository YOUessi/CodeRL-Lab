"""EXP-014B single-model AI blind review aggregation.

This is NOT a human study and does not create fake rater identities.
The 64 blind judgments and their SHA were committed BEFORE this script
can access the separate A/B/C-to-model key. It honors unreviewed, no
acceptable response, equal-best and limited safety categories.
Only aggregate scores and cryptographic provenance are shared in GitHub.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ARMS=("sft","bf16_dpo","fp32_dpo")
LETTERS=("A","B","C")
EXPECTED_SOURCE_SHA="29eff706811eee7cdf7f095469242f43a3c58b28c0c13239af1ebe40b08955da"
EXPECTED_RATINGS_SHA="b8faab1d58134dcaf51769661ab0ff0cd54924f62e3ea44c602a2af54f62df88"
EXPECTED_KEY_SHA="122ec55371324e4ce380efd791ef41898b1b161270a5ddfb7370950a7a65fe96"


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def read_lines(path:Path)->list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def verify_before_unblind(*, packets:Path, ratings:Path, commitment:Path)->tuple[list[dict],list[dict],dict]:
    freeze=json.loads(commitment.read_text(encoding="utf-8"))
    if (freeze.get("status")!="ratings_frozen_before_unblinding"
        or freeze.get("independent_human_reviewers")!=0
        or freeze.get("human_review_completed") is not False
        or freeze.get("source_blind_packet_sha256")!=EXPECTED_SOURCE_SHA
        or freeze.get("blind_ratings_file_sha256")!=EXPECTED_RATINGS_SHA):
        raise ValueError("missing PRE-KEY SHA-committed single-AI review identity")
    if sha256(packets)!=EXPECTED_SOURCE_SHA or sha256(ratings)!=EXPECTED_RATINGS_SHA:
        raise ValueError("blinded packet or precommitted AI ratings was modified")
    packet_rows=read_lines(packets)
    rated_rows=read_lines(ratings)
    if len(packet_rows)!=64 or len(rated_rows)!=64:
        raise ValueError("must have all 64 frozen source prompts and 64 judgments")
    status_counts=Counter()
    for i,(question,rating) in enumerate(zip(packet_rows,rated_rows)):
        if rating.get("index")!=i or rating.get("kind")!="AI_SINGLE_REVIEWER_BLIND":
            raise ValueError("AI source rating index/type corrupted")
        if rating.get("source_packet_sha256")!=EXPECTED_SOURCE_SHA:
            raise ValueError("reviewed a different prompt-only source")
        if set(rating.get("scores",{}))!=set(LETTERS):
            raise ValueError("each question needs A/B/C scores")
        if not isinstance(rating.get("reason_zh"),str) or not rating["reason_zh"].strip():
            raise ValueError("review lacked per-task written explanation")
        top=rating.get("top_labels")
        status=rating.get("status")
        if (not isinstance(top,list) or len(top)!=len(set(top))
            or any(t not in LETTERS for t in top)):
            raise ValueError("invalid tied-best labels")
        scores=rating["scores"]
        if status=="limited_safety":
            if top or any(scores[t] is not None for t in LETTERS):
                raise ValueError("cannot award grade to restricted unreviewed item")
        else:
            if any(type(scores[t])!=int or not 0<=scores[t]<=4 for t in LETTERS):
                raise ValueError("all reviewed A/B/C scores must be 0..4")
            max_score=max(scores.values())
            if status=="no_acceptable_answer":
                if top or max_score>1:
                    raise ValueError("no acceptable response must not have winner")
            elif status in ("winner","tied"):
                if top!=[t for t in LETTERS if scores[t]==max_score]:
                    raise ValueError("top labels do not match maximal grades")
                if (status=="winner")!=(len(top)==1):
                    raise ValueError("tie/winner status wrong")
            else:
                raise ValueError(f"unknown AI status {status}")
        status_counts[status]+=1
        if (not question.get("task_id") or
            set(question.get("candidate_answers",{}))!=set(LETTERS)):
            raise ValueError("incomplete blinded candidate answer")
    if dict(status_counts)!={k:freeze[k] for k in
            ("single_clear_winner","tied","no_acceptable_response","safety_read_restricted")
            for k in []}:
        pass
    expected={
        "winner":int(freeze["single_clear_winner"]),
        "tied":int(freeze["tied"]),
        "no_acceptable_answer":int(freeze["no_acceptable_response"]),
        "limited_safety":int(freeze["safety_read_restricted"]),
    }
    if dict(status_counts)!=expected:
        raise ValueError("AI pre-unblind category counts changed")
    return packet_rows,rated_rows,freeze


def aggregate(*, packet_rows:list[dict],reviews:list[dict],keys:list[dict],keysha:str,freeze:dict)->dict:
    if len(keys)!=64:
        raise ValueError("must have all64 blind assignment keys")
    if [r["task_id"] for r in keys]!=[r["task_id"] for r in packet_rows]:
        raise ValueError("AI review task order does not match the frozen model-key file")
    by_model={arm:{
        "graded_answers":0,"grade_sum":0,"grade4_count":0,"grade0_count":0,
        "clear_wins":0,"partial_tied_win_credits":0.0,
        "tied_top_membership":0,
    } for arm in ARMS}
    pairwise={}
    for i,a in enumerate(ARMS):
        for b in ARMS[i+1:]:
            pairwise[a+"_vs_"+b]={"first_model_higher":0,"second_model_higher":0,"equal_grade":0}
    for idx,(r,key,row) in enumerate(zip(reviews,keys,packet_rows)):
        mapping=key.get("mapping",{})
        if set(mapping)!=set(LETTERS) or set(mapping.values())!=set(ARMS):
            raise ValueError("unblinding key contains duplicate or unknown model")
        score=r["scores"]
        for letter,arm in mapping.items():
            if (hashlib.sha256(row["candidate_answers"][letter].encode()).hexdigest()
                != key["answer_sha256"][letter]):
                raise ValueError("blinded answer changed after rating was frozen")
            if r["status"]!="limited_safety":
                model=by_model[arm]
                model["graded_answers"]+=1
                model["grade_sum"]+=score[letter]
                model["grade4_count"]+=int(score[letter]==4)
                model["grade0_count"]+=int(score[letter]==0)
                if r["status"]=="winner" and letter in r["top_labels"]:
                    model["clear_wins"]+=1
                if r["status"]=="tied" and letter in r["top_labels"]:
                    model["partial_tied_win_credits"]+=1/len(r["top_labels"])
                    model["tied_top_membership"]+=1
        if r["status"]=="limited_safety":
            continue
        inverse={arm:letter for letter,arm in mapping.items()}
        for pair,result in pairwise.items():
            a,b=pair.split("_vs_")
            left,right=score[inverse[a]],score[inverse[b]]
            label="first_model_higher" if left>right else (
                "second_model_higher" if right>left else "equal_grade"
            )
            result[label]+=1
    for arm,item in by_model.items():
        item["mean_grade_0_to_4"]=item["grade_sum"]/item["graded_answers"]
        item["win_credit_clear_plus_fractional_ties"]=(
            item["clear_wins"]+item["partial_tied_win_credits"]
        )
    return {
        "experiment":"EXP-014B-SINGLE-AI-BLIND-PROXY",
        "status":"completed_ai_proxy_not_human_review",
        "method":"single_GPT6_review_with_helpfulness_and_fact_safety_rubrics",
        "human_reviewers":0,
        "independent_human_review_completed":False,
        "number_of_AI_assessors":1,
        "frozen_tasks":64,"candidates_per_task":3,
        "graded_tasks":63,
        "graded_answers":63*3,
        "no_clear_winner":int(freeze["tied"])+int(freeze["no_acceptable_response"]),
        "tie_tasks":int(freeze["tied"]),
        "all_answers_unacceptable_tasks":int(freeze["no_acceptable_response"]),
        "safety_restricted_tasks":int(freeze["safety_read_restricted"]),
        "clear_winner_tasks":int(freeze["single_clear_winner"]),
        "by_arm":by_model,
        "pairwise_grade_comparison":pairwise,
        "frozen_blind_packet_sha256":EXPECTED_SOURCE_SHA,
        "AI_judgments_SHA256_frozen_before_model_key":EXPECTED_RATINGS_SHA,
        "separate_key_file_sha256":keysha,
        "commitment_manifest_sha256":None,
        "review_files_local_only":True,
        "generated_response_text_exported":False,
        "disclaimer":"This is ONE AI model's blinded proxy review. Two rubric checks by one AI are NOT equivalent to two independent human reviewers; no human interrater agreement may be claimed. Grades are subjective comparative judgments, not benchmark correctness.",
        "method_limitations":"Public historical HH prompts; response provenance may be known to pretrained models; no independent human ratings, no interrater reliability, one safety-restricted item and five all-unacceptable cases.",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--packets",type=Path,required=True)
    p.add_argument("--ratings",type=Path,required=True)
    p.add_argument("--commitment",type=Path,required=True)
    p.add_argument("--key",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    p_rows,judgements,freeze=verify_before_unblind(
        packets=args.packets,ratings=args.ratings,commitment=args.commitment
    )
    # The separate unblinding key is FIRST opened after frozen-review validation.
    if sha256(args.key)!=EXPECTED_KEY_SHA:
        raise ValueError("blind assignment key checksum differs from the generated corpus")
    keyrows=read_lines(args.key)
    result=aggregate(packet_rows=p_rows,reviews=judgements,keys=keyrows,
                     keysha=EXPECTED_KEY_SHA,freeze=freeze)
    result["commitment_manifest_sha256"]=sha256(args.commitment)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():
        if json.loads(args.output.read_text())!=result:
            raise FileExistsError("refusing changed AI review report")
    else:
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
