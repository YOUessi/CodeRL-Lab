"""EXP-014B: descriptive paired statistics on one AI reviewer's frozen anonymous grades.
This is NOT two human raters or independent benchmark validation.
"""
import argparse, json, random, math, hashlib
from pathlib import Path
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def pval(a,b):
    n=a+b
    return min(1., 2*sum(math.comb(n,j) for j in range(min(a,b)+1))/2**n) if n else 1.
def run(rating_path, key_path, frozen_path):
    frozen=json.loads(frozen_path.read_text())
    expect={"rating":"b8faab1d58134dcaf51769661ab0ff0cd54924f62e3ea44c602a2af54f62df88",
            "key":"122ec55371324e4ce380efd791ef41898b1b161270a5ddfb7370950a7a65fe96"}
    if sha(rating_path)!=expect["rating"] or sha(key_path)!=expect["key"]:
        raise ValueError("frozen rating/key mismatch")
    if frozen["status"]!="completed_ai_proxy_not_human_review" or frozen["human_reviewers"]!=0:
        raise ValueError("not one AI blind reviewer result")
    rated=[json.loads(x) for x in rating_path.read_text().splitlines() if x.strip()]
    keys=[json.loads(x) for x in key_path.read_text().splitlines() if x.strip()]
    rows=[]
    for i,(r,k) in enumerate(zip(rated,keys)):
        if r["index"]!=i:raise ValueError("order mismatch")
        if r["status"]=="limited_safety":continue
        rows.append({arm:r["scores"][letter] for letter,arm in k["mapping"].items()})
    if len(rows)!=63:raise ValueError("expected 63 graded task triplets")
    result={}
    for a,b in [("sft","bf16_dpo"),("sft","fp32_dpo"),("fp32_dpo","bf16_dpo")]:
        delta=[r[a]-r[b] for r in rows]
        wins=sum(x>0 for x in delta);losses=sum(x<0 for x in delta)
        rng=random.Random(42);n=len(delta)
        v=sorted(sum(delta[rng.randrange(n)] for j in range(n))/n for i in range(20000))
        result[a+"_minus_"+b]={"paired_graded_questions":n,"first_higher":wins,"second_higher":losses,
              "equal":n-wins-losses,"mean_ordinal_grade_difference":sum(delta)/n,
              "task_bootstrap_ci95":[v[500],v[19499]],
              "unadjusted_exact_sign_test_p":pval(wins,losses)}
    return {"experiment":"EXP-014B-AI-RATER-PAIRWISE-DESCRIPTIVE","status":"completed",
            "reviewers":"ONE AI; human reviewers zero","grade_pairs":63,
            "comparison":result,"blind_rating_sha256":expect["rating"],"identity_sha256":expect["key"],
            "limitations":"Single AI subjective 0-4 ordinal grades. Intervals and sign tests are posthoc descriptive, unadjusted for 3 comparisons. Not human validation."}
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for arg in ["ratings","key","aggregate","output"]:p.add_argument("--"+arg,type=Path,required=True)
    a=p.parse_args();out=run(a.ratings,a.key,a.aggregate)
    if a.output.exists():raise FileExistsError("immutable report")
    a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(out,ensure_ascii=False,indent=2))
