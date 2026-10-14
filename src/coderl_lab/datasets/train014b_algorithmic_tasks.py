"""TRAIN-014B algorithmic Python code tasks; test oracles frozen before GPU.

36 training specs and 12 disjoint heldout specs, 6 algorithm families. Model
sees only instructions; GRPO reward callback sees frozen train tests; heldout
test labels remain inaccessible until post-training verification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
REVISION="989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
TRAIN={
    "pairs":[-9,-4,0,3,6,11],
    "subarray_mod":[2,3,4,5,7,9],
    "runs_ge":[1,2,3,4,5,6],
    "clusters_above":[-3,0,1,2,5,7],
    "best_subarray":[0,1,3,5,7,9],
    "distinct_span":[0,1,3,5,7,9],
}
HELDOUT={
    "pairs":[-6,8],
    "subarray_mod":[6,8],
    "runs_ge":[7,8],
    "clusters_above":[-5,4],
    "best_subarray":[2,6],
    "distinct_span":[2,4],
}
ARRS=[
    [],[0],[1],[-1],[2,2],[1,2,3],[3,2,1],[0,0,0],
    [-5,-1,-2],[5,-2,3],[-3,2,-3,2],[-3,-2,-1],[4,4,-1,4],
    [2,-2,2,-2,2],[0,1,0,1,0],[-9,7,5,-3,8],[9,-1,-2,-3,9],
    [1,7,-5,3,-2,7,1],[4,-3,4,-3,4,-3,4],
]
STRINGS=["","a","aa","aba","abc","abca","aabb","abcdef",
         "pwwkew","abba","dvdf","abcabcbb","aaaaab","tmmzuxt",
         "longest","ababab","xyzzxy","a!b!c","a1a2a3a4"]
def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda:f.read(1048576),b""):h.update(part)
    return h.hexdigest()

def expected(kind:str,param:int,x)->int:
    if kind=="pairs":
        return sum(1 for i in range(len(x)) for j in range(i+1,len(x)) if x[i]+x[j]==param)
    if kind=="subarray_mod":
        if param<=0:raise ValueError("positive modulus only")
        return sum(1 for i in range(len(x)) for j in range(i+1,len(x)+1) if sum(x[i:j])%param==0)
    if kind=="runs_ge":
        if not x:return 0
        best=run=1
        for i in range(1,len(x)):
            run=run+1 if x[i]-x[i-1]>=param else 1
            best=max(best,run)
        return best
    if kind=="clusters_above":
        return sum(1 for i,v in enumerate(x) if v>param and (i==0 or x[i-1]<=param))
    if kind=="best_subarray":
        if not x:return 0
        best=running=x[0]
        for value in x[1:]:
            running=max(value,running+value)
            best=max(best,running)
        return best-param
    if kind=="distinct_span":
        longest=0
        for i in range(len(x)):
            used=set()
            for c in x[i:]:
                if c in used:break
                used.add(c)
            longest=max(longest,len(used))
        return longest+param
    raise ValueError("unsupported task")

def problem(kind:str,param:int)->str:
    action={
        "pairs":f"For a list of integers x, return how many index pairs i<j have x[i]+x[j] equal to {param}. Count index pairs, not distinct values.",
        "subarray_mod":f"For a list of integers x, return the number of nonempty contiguous subarrays whose sum is divisible by {param}. Negative numbers are allowed.",
        "runs_ge":f"For a list of integers x, return the maximum number of elements in a contiguous run in which every adjacent difference (next minus previous) is at least {param}. Empty list returns 0.",
        "clusters_above":f"For a list of integers x, count maximal nonempty contiguous runs of numbers strictly greater than {param}. Empty list returns 0.",
        "best_subarray":f"For a list of integers x, find the maximum sum among all nonempty contiguous subarrays, then subtract {param}. For an empty input return 0. Negative numbers are allowed.",
        "distinct_span":f"For a string x, find the maximum length of a contiguous substring with no repeated characters, then add {param}. This also applies to the empty string.",
    }[kind]
    return (
        "Implement Python code defining a top-level function solve(x) that "
        "returns an integer. Output only executable Python source code, "
        "not an explanation or Markdown. " + action
    )

def inputs(kind:str,param:int)->list:
    if kind=="distinct_span":return list(STRINGS)
    rng=random.Random(42+int(hashlib.sha256((kind+str(param)).encode()).hexdigest()[:8],16))
    cases=list(ARRS)
    for _ in range(7):
        cases.append([rng.randint(-9,9) for i in range(rng.randint(0,9))])
    seen=set()
    unique=[]
    for value in cases:
        fingerprint=json.dumps(value)
        if fingerprint in seen:continue
        seen.add(fingerprint)
        unique.append(value)
    return unique

def build(specs:dict, split:str)->list[dict]:
    rows=[]
    for kind,values in sorted(specs.items()):
        for n in values:
            qid=f"{split}/{kind}/{n}"
            tests=[{"x":x,"expected":expected(kind,n,x)} for x in inputs(kind,n)]
            rows.append({
                "task_id":qid,
                "task_kind":kind,
                "prompt":[{"role":"user","content":problem(kind,n)}],
                "test_cases":tests,
            })
    return rows

def freeze(out:Path)->dict:
    out.mkdir(parents=True,exist_ok=True)
    train=build(TRAIN,"train")
    heldout=build(HELDOUT,"heldout")
    if len(train)!=36 or len(heldout)!=12:raise ValueError("data count drift")
    if ({(r["task_kind"],r["task_id"].split("/")[-1]) for r in train} &
        {(r["task_kind"],r["task_id"].split("/")[-1]) for r in heldout}):
        raise ValueError("heldout task params overlap training")
    hold_prompt=[{k:r[k] for k in ("task_id","task_kind","prompt")} for r in heldout]
    hold_oracle=[{k:r[k] for k in ("task_id","test_cases")} for r in heldout]
    paths={"train":out/"train.jsonl",
           "holdout_public":out/"heldout_public.jsonl",
           "holdout_oracle":out/"heldout_oracle.jsonl"}
    for name,rows in (("train",train),("holdout_public",hold_prompt),("holdout_oracle",hold_oracle)):
        with paths[name].open("x",encoding="utf-8") as f:
            for row in rows:f.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+"\n")
    summary={
        "experiment":"TRAIN-014B-ALGORITHMIC-GRPO",
        "source_script_sha256":sha(Path(__file__)),
        "model":MODEL,"model_revision":REVISION,
        "seed":42,"train_task_count":36,"heldout_task_count":12,
        "train_heldout_spec_overlap":0,
        "task_family_count":6,
        "train_oracle_only_in_reward_metadata":True,
        "heldout_oracle_never_seen_by_GRPOTrainer":True,
        "sha256":{k:sha(v) for k,v in paths.items()},
        "source_intent":"mid-difficulty deterministic array/string algorithms; exploratory toy tasks, not production coding benchmark",
    }
    (out/"manifest.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",type=Path,required=True)
    print(json.dumps(freeze(p.parse_args().output_dir),indent=2))
