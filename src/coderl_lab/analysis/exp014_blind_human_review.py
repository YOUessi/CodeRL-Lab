"""EXP-014 blind human response review; NO automatic model-as-judge labels.

Reviewers receive only 64 public prompts and shuffled A/B/C responses.
They do not see which model produced a response. Do not unblind scores
until two independent completed review JSONL files are available.
All original text stays on Tang, only aggregate counts may enter GitHub.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


WINNERS=("A","B","C","TIE")


def sha_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path:Path)->list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def get_packet_ids(path:Path, expected:int=64)->set[str]:
    rows=read_jsonl(path)
    ids=set()
    for row in rows:
        if set(row["candidate_answers"])!={"A","B","C"}:
            raise ValueError("missing randomized A/B/C responses")
        if row["task_id"] in ids:
            raise ValueError("duplicate packet task")
        ids.add(row["task_id"])
    if len(ids)!=expected:
        raise ValueError(f"review pack must contain {expected} distinct tasks")
    return ids


def read_completed_rater(path:Path, ids:set[str])->tuple[str,dict]:
    responses=read_jsonl(path)
    if len(responses)!=len(ids):
        raise ValueError("all frozen prompts must be independently reviewed")
    rater_names={r.get("rater_id") for r in responses}
    if len(rater_names)!=1 or not next(iter(rater_names)):
        raise ValueError("each reviewer file must have one non-empty rater ID")
    name=next(iter(rater_names))
    by_task={}
    for row in responses:
        key=row.get("task_id")
        if key in by_task or key not in ids:
            raise ValueError("duplicate or invented review task")
        if row.get("winner") not in WINNERS:
            raise ValueError("winner must be A/B/C/TIE")
        by_task[key]=row["winner"]
    return name,by_task


def blinded_agreement(left:dict,right:dict)->dict:
    ids=sorted(left)
    if not ids or set(right)!=set(left):
        raise ValueError("reviewers must cover same task IDs")
    joint=Counter((left[q],right[q]) for q in ids)
    observed=sum(left[q]==right[q] for q in ids)/len(ids)
    a=Counter(left.values());b=Counter(right.values())
    expected=sum(a[cat]*b[cat] for cat in WINNERS)/len(ids)**2
    kappa=(observed-expected)/(1-expected) if expected<1 else float("nan")
    return {"tasks":len(ids),"raw_agreement":observed,
            "cohen_kappa_unweighted":kappa if math.isfinite(kappa) else None,
            "tie_votes":a["TIE"]+b["TIE"],
            "disagreements":sum(left[q]!=right[q] for q in ids)}


def aggregate(packets:Path,key_file:Path,rater_a:Path,rater_b:Path,expected:int=64)->dict:
    ids=get_packet_ids(packets,expected=expected)
    ra,votes_a=read_completed_rater(rater_a,ids)
    rb,votes_b=read_completed_rater(rater_b,ids)
    if ra==rb:
        raise ValueError("two independent reviewers must have distinct IDs")
    rows=read_jsonl(key_file)
    keys={}
    for row in rows:
        qid=row["task_id"]
        if qid in keys or set(row["mapping"])!={"A","B","C"} or set(row["mapping"].values())!={"sft","bf16_dpo","fp32_dpo"}:
            raise ValueError("invalid blinded mapping")
        keys[qid]=row["mapping"]
    if set(keys)!=ids:
        raise ValueError("blind mapping tasks do not correspond to ratings")
    agreement=blinded_agreement(votes_a,votes_b)
    votes=Counter()
    consensus=Counter()
    for qid in sorted(ids):
        for win in (votes_a[qid],votes_b[qid]):
            if win!="TIE":
                votes[keys[qid][win]]+=1
        if votes_a[qid]==votes_b[qid] and votes_a[qid]!="TIE":
            consensus[keys[qid][votes_a[qid]]]+=1
    return {
        "experiment":"EXP-014-REAL-HUMAN-BLIND-REVIEW",
        "status":"two_human_reviewers_completed",
        "task_count":expected,
        "reviewer_count":2,
        "blind_packet_sha256":sha_file(packets),
        "model_assignment_key_sha256":sha_file(key_file),
        "human_agreement":agreement,
        "human_winner_votes_by_model":dict(votes),
        "unanimous_wins_by_model":dict(consensus),
        "total_winner_votes":sum(votes.values()),
        "limitations":"Small historical helpfulness sample; two reviewers, not a forward independent trial. Tie votes excluded from model wins. Rater IDs/individual ratings are not published.",
    }


def review(packets:Path,who:str,output:Path,limit:int|None=None):
    if not who.strip():
        raise ValueError("non-empty rater id required")
    if output.exists():
        raise FileExistsError("review file already exists; refusing overwrite")
    rows=read_jsonl(packets)
    if limit is not None:rows=rows[:limit]
    with output.open("x",encoding="utf-8") as f:
        for i,item in enumerate(rows,1):
            print(f"\n--- Question {i}/{len(rows)}: {item['task_id']} ---")
            for msg in item["messages"]:
                print(f"\n{msg['role']}: {msg['content']}\n")
            for letter,answer in item["candidate_answers"].items():
                print(f"\n--- {letter} ---\n{answer}\n")
            while True:
                response=input("Choose A/B/C/TIE (no model names given): ").strip().upper()
                if response in WINNERS:break
            f.write(json.dumps({"task_id":item["task_id"],
                                "rater_id":who,"winner":response})+"\n")
            f.flush()
    print(f"Review complete; ratings stored privately: {output}")


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="action",required=True)
    a=sub.add_parser("review")
    a.add_argument("--packets",type=Path,required=True)
    a.add_argument("--rater-id",required=True)
    a.add_argument("--output",type=Path,required=True)
    a.add_argument("--max-tasks",type=int)
    b=sub.add_parser("aggregate")
    b.add_argument("--packets",type=Path,required=True)
    b.add_argument("--key",type=Path,required=True)
    b.add_argument("--rater-a",type=Path,required=True)
    b.add_argument("--rater-b",type=Path,required=True)
    b.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.action=="review":
        review(args.packets,args.rater_id,args.output,args.max_tasks)
    else:
        stats=aggregate(args.packets,args.key,args.rater_a,args.rater_b)
        if args.output.exists():
            raise FileExistsError("unblinded human quality report already exists")
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(stats,indent=2)+"\n")
        print(json.dumps(stats,indent=2))

if __name__=="__main__":main()
