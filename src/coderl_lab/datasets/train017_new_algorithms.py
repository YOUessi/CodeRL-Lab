"""TRAIN-017 new 7B GRPO algorithmic data, before any model training.

48 training specs and 24 heldout specs, six previously unused algorithm
families. Heldout oracle is stored separately; generation reads prompts ONLY.
Every test is deterministically calculated from a pure reference algorithm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

EXPERIMENT="TRAIN-017-7B-NEW-CODE-GRPO"
MODEL="Qwen/Qwen2.5-7B-Instruct"
REVISION="a09a35458c702b33eeacc393d103063234e8bc28"
SEED=20261015
TRAIN_PARAMS={
    "gcd_target":[1,2,3,4,6,8,9,12],
    "hist_shift":[-6,-4,-2,0,1,3,5,7],
    "jump_cap":[1,2,3,4,5,7,9,11],
    "pal_del_le":[0,1,2,3,5,7,9,11],
    "exact_k_chars":[1,3,5,7,9,11,13,15],
    "xor_target":[0,1,2,3,5,7,9,11],
}
HOLD_PARAMS={
    "gcd_target":[5,7,10,14],
    "hist_shift":[-5,-3,2,6],
    "jump_cap":[6,8,10,12],
    "pal_del_le":[4,6,8,10],
    "exact_k_chars":[2,4,6,8],
    "xor_target":[4,6,8,10],
}
LETTERS="abcdefghijklmnopqrstu"
GENERATOR_VERSION="2026-10-15-new-algorithmic-families-v1"


def digest(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""):
            h.update(block)
    return h.hexdigest()


def oracle(kind:str,k:int,x)->int:
    if kind=="gcd_target":
        count=0
        for left in range(len(x)):
            g=0
            for val in x[left:]:
                g=math.gcd(g,abs(val))
                count+=int(g==k)
        return count
    if kind=="hist_shift":
        h=[max(0,y+k) for y in x]
        best=0
        for i in range(len(h)):
            low=10**15
            for j in range(i,len(h)):
                low=min(low,h[j])
                best=max(best,low*(j-i+1))
        return best
    if kind=="jump_cap":
        if not x:return -1
        if len(x)==1:return 0
        dp=[10**8]*len(x)
        dp[0]=0
        for i,val in enumerate(x):
            if dp[i]==10**8:continue
            step=min(max(0,val),k)
            for j in range(i+1,min(len(x),i+step+1)):
                dp[j]=min(dp[j],dp[i]+1)
        return dp[-1] if dp[-1]!=10**8 else -1
    if kind=="pal_del_le":
        if not x:return 1
        n=len(x)
        dp=[[0]*n for _ in range(n)]
        for i in range(n):dp[i][i]=1
        for width in range(2,n+1):
            for i in range(n-width+1):
                j=i+width-1
                dp[i][j]=(2+(dp[i+1][j-1] if width>2 else 0)
                          if x[i]==x[j] else max(dp[i+1][j],dp[i][j-1]))
        return int((n-dp[0][-1])<=k)
    if kind=="exact_k_chars":
        total=0
        for i in range(len(x)):
            seen={}
            for c in x[i:]:
                seen[c]=seen.get(c,0)+1
                if len(seen)==k:total+=1
                if len(seen)>k:break
        return total
    if kind=="xor_target":
        total=0
        for i in range(len(x)):
            xor=0
            for val in x[i:]:
                xor ^= val
                total+=int(xor==k)
        return total
    raise ValueError("unknown immutable family")


def prompt(kind:str,k:int)->str:
    specs={
        "gcd_target":f"For an integer list x, count nonempty contiguous subarrays whose greatest common divisor of absolute values is exactly {k}. The gcd of all-zero subarrays is 0. Return 0 on an empty list.",
        "hist_shift":f"For an integer list x, first turn each element v into height max(0, v + ({k})). Return the maximum area of a contiguous rectangle under this histogram (integer bar widths 1). For an empty list return 0.",
        "jump_cap":f"For a list x of nonnegative jump capacities, start at index 0 and try to reach the final index. From position i, one jump can move forward by 1 through min(x[i], {k}) indices. Return the minimum number of jumps, or -1 if unreachable. Empty list returns -1 and one element returns 0.",
        "pal_del_le":f"For an ASCII string x, return integer 1 if it is possible to make the whole string a palindrome by deleting at most {k} characters (without rearranging survivors); otherwise return 0. Empty string is already palindromic.",
        "exact_k_chars":f"For a lowercase ASCII string x, return the number of nonempty contiguous substrings that contain exactly {k} distinct characters. Count by starting and ending positions; empty string returns 0.",
        "xor_target":f"For a list of nonnegative integers x, return the number of nonempty contiguous subarrays whose bitwise XOR is exactly {k}. Count subarrays by indices; empty list returns 0.",
    }
    return "Write Python source code only; define a top-level function solve(x) and return an integer. Do not include explanations or Markdown fences. "+specs[kind]


def examples(kind:str,k:int,split:str)->list:
    seed=int(hashlib.sha256(f"{GENERATOR_VERSION}|{kind}|{k}|{split}".encode()).hexdigest()[:16],16)
    rng=random.Random(seed)
    if kind in ("pal_del_le","exact_k_chars"):
        defaults=[
            "","a","aa","aba","abba","abc","abca","abcabc",
            "abracadabra","a"*18,"ab"*12,"abc"*8,"abcdefghijklmno",
            "abcdefghijklmnopqrstu","abcdefghijklmnoabcde",
            "abcdefghijklmnopqrstuabc","civic","aabbccddeeff",
            "xyzzyx","noonabcnoon","abcddcba","".join(reversed(LETTERS)),
            "aaabbbbccccddddeeeefff","abbccddeeffgghhiijjkk",
        ]
        values=list(defaults)
        for i in range(10):
            n=rng.randint(9,27)
            alphabet=LETTERS[:rng.randint(2,20)]
            values.append("".join(rng.choice(alphabet) for _ in range(n)))
    else:
        values=[
            [],[0],[1],[-1],[2,2],[1,2,3],[3,2,1],[0,0,0],
            [-5,-1,-2],[5,-2,3],[-3,2,-3,2],[-3,-2,-1],[4,4,-1,4],
            [2,-2,2,-2,2],[0,1,0,1,0],[-9,7,5,-3,8],
            [9,-1,-2,-3,9],[1,7,-5,3,-2,7,1],[4,-3,4,-3,4,-3,4],
            [-12,6,-6,12],[-8,-4,0,4,8],[7]*8,[1]*13,
        ]
        if kind=="jump_cap":
            values=[list(map(lambda t:max(0,t),v)) for v in values]
            values += [[1]*15,[2]*15,[3]*15,[0,0,1,0,1,0],
                       [12]+[0]*13,[1,0,12]+[0]*12,[4,3,0,2,0,9,0]]
        if kind=="xor_target":
            values=[[max(0,z)&31 for z in v] for v in values]
            values += [[k,k],[k,0,k],[0,k,0,k],list(range(16))]
        if kind=="gcd_target":
            values += [[k],[-k],[k,k],[2*k,3*k,k],[0,k,0],[-k,2*k,-2*k],[0,0,k]]
        if kind=="hist_shift":
            values += [[-k],[-k,-k],[1-k]*8,[5-k]*7,[0-k,5-k,0-k]]
        for i in range(12):
            n=rng.randint(4,18)
            if kind=="jump_cap":
                sample=[rng.randrange(0,14) for _ in range(n)]
                if i%4==0:sample[0]=0
            elif kind=="xor_target":
                sample=[rng.randrange(0,32) for _ in range(n)]
            elif kind=="gcd_target":
                sample=[rng.choice([0,k,-k,2*k,3*k,rng.randint(-18,18)]) for _ in range(n)]
            else:
                sample=[rng.randint(-9,16) for _ in range(n)]
            values.append(sample)
    # Deduplicate inputs but preserve seeded order; strict sandbox test budget 10..40
    seen=set();unique=[]
    for v in values:
        key=json.dumps(v,sort_keys=True)
        if key in seen:continue
        seen.add(key);unique.append(v)
    return unique[:40]


def create(specs:dict,split:str)->list[dict]:
    out=[]
    for kind,values in sorted(specs.items()):
        for k in values:
            cases=[{"x":x,"expected":oracle(kind,k,x)} for x in examples(kind,k,split)]
            if not 20<=len(cases)<=40:
                raise ValueError(f"too few or too many test cases for {kind}/{k}: {len(cases)}")
            out.append({
                "task_id":f"{split}/{kind}/{k}",
                "task_kind":kind,
                "prompt":[{"role":"user","content":prompt(kind,k)}],
                "test_cases":cases,
            })
    return out


def freeze(root:Path)->dict:
    if root.exists():
        raise FileExistsError("frozen dataset dir already exists; never overwrite")
    root.mkdir(parents=True)
    train=create(TRAIN_PARAMS,"train")
    heldout=create(HOLD_PARAMS,"heldout")
    if len(train)!=48 or len(heldout)!=24:
        raise ValueError("unexpected task count")
    specs_train={(r["task_kind"],r["task_id"].rsplit("/",1)[-1]) for r in train}
    specs_hold={(r["task_kind"],r["task_id"].rsplit("/",1)[-1]) for r in heldout}
    if specs_train & specs_hold:
        raise ValueError("train/holdout parameter overlap")
    public=[{k:r[k] for k in ("task_id","task_kind","prompt")} for r in heldout]
    oracle_only=[{k:r[k] for k in ("task_id","test_cases")} for r in heldout]
    paths={"train":root/"train.jsonl","holdout_public":root/"holdout_public.jsonl",
           "holdout_oracle":root/"holdout_oracle.jsonl"}
    for label,records in (
        ("train",train),("holdout_public",public),("holdout_oracle",oracle_only)
    ):
        with paths[label].open("x",encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+"\n")
    result={
        "experiment":EXPERIMENT,"source_script_sha256":digest(Path(__file__)),
        "model":MODEL,"model_revision":REVISION,"seed":SEED,
        "train_tasks":48,"heldout_tasks":24,"task_families":sorted(TRAIN_PARAMS),
        "samples_per_task_eval":4,"evaluation_seeds":[42,314159],
        "training_oracle_not_in_prompts":True,
        "heldout_oracle_never_seen_by_training":True,
        "train_holdout_parameter_overlap":0,
        "generated_before_training_or_model_evaluation":True,
        "sha256":{k:digest(v) for k,v in paths.items()},
        "disclaimer":"Fresh within CodeRL-Lab deterministic synthetic algorithmic tasks, not novel mathematical problems nor a guarantee against pretrained model familiarity."
    }
    (root/"manifest.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True,type=Path)
    print(json.dumps(freeze(ap.parse_args().out),indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()
