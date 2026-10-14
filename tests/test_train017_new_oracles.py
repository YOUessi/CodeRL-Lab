"""Independently verify TRAIN-017 reference labels and frozen data separation."""
from functools import lru_cache
from collections import Counter
import math

from coderl_lab.datasets.train017_new_algorithms import (
    TRAIN_PARAMS, HOLD_PARAMS, oracle,create,examples,
)


def independent(kind,k,x):
    if kind=="gcd_target":
        total=0
        counts={}
        for z in x:
            fresh={abs(z):1}
            for val,qty in counts.items():
                g=math.gcd(val,abs(z))
                fresh[g]=fresh.get(g,0)+qty
            counts=fresh
            total+=counts.get(k,0)
        return total
    if kind=="hist_shift":
        heights=[max(0,v+k) for v in x]
        stack=[]
        answer=0
        for j,h in enumerate(heights+[0]):
            while stack and heights[stack[-1]]>h:
                old=stack.pop()
                start=stack[-1]+1 if stack else 0
                answer=max(answer,heights[old]*(j-start))
            stack.append(j)
        return answer
    if kind=="jump_cap":
        if not x:return -1
        distance={0:0}
        queue=[0]
        for idx in queue:
            for nxt in range(idx+1,min(len(x),idx+min(x[idx],k)+1)):
                if nxt not in distance:
                    distance[nxt]=distance[idx]+1
                    queue.append(nxt)
        return distance.get(len(x)-1,-1)
    if kind=="pal_del_le":
        @lru_cache(None)
        def deletions(a,b):
            if a>=b:return 0
            if x[a]==x[b]:return deletions(a+1,b-1)
            return 1+min(deletions(a+1,b),deletions(a,b-1))
        return int(deletions(0,len(x)-1)<=k)
    if kind=="exact_k_chars":
        def at_most(target):
            if target<=0:return 0
            left=0;counts=Counter();total=0
            for right,c in enumerate(x):
                counts[c]+=1
                while len(counts)>target:
                    ch=x[left];counts[ch]-=1
                    if not counts[ch]:del counts[ch]
                    left+=1
                total+=right-left+1
            return total
        return at_most(k)-at_most(k-1)
    if kind=="xor_target":
        prefix=0;count=Counter({0:1});total=0
        for val in x:
            prefix ^= val
            total+=count[prefix ^ k]
            count[prefix]+=1
        return total
    raise ValueError(kind)


def test_six_independent_algorithm_oracles_cover_all_frozen_tasks():
    train=create(TRAIN_PARAMS,"train")
    heldout=create(HOLD_PARAMS,"heldout")
    assert len(train)==48 and len(heldout)==24
    seen=set()
    for row in train+heldout:
        assert row["task_id"] not in seen
        seen.add(row["task_id"])
        kind=row["task_kind"]
        k=int(row["task_id"].rsplit("/",1)[1])
        assert 20<=len(row["test_cases"])<=40
        assert set(row["prompt"][0])=={"role","content"}
        for case in row["test_cases"]:
            assert independent(kind,k,case["x"])==case["expected"],row["task_id"]
        assert "test_cases" not in str(row["prompt"])


def test_parameter_disjoint_and_new_task_family_ids():
    for kind in TRAIN_PARAMS:
        assert set(TRAIN_PARAMS[kind]).isdisjoint(HOLD_PARAMS[kind])
    old={"pairs","subarray_mod","runs_ge","clusters_above",
         "best_subarray","distinct_span"}
    assert not set(TRAIN_PARAMS)&old
