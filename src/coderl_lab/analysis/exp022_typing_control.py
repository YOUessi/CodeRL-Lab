"""EXP-022 controlled Docker-only typing.List import diagnostic.

Operates only on already frozen, scored regression pairs. Never updates the
benchmark score or modifies original candidates.
"""
import argparse,hashlib,json,re
from pathlib import Path
from coderl_lab.analysis.exp021_seed42_docker_score import extract_code,run_docker,verify_image

def rows(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def run(root:Path,audit:Path):
    flips=json.loads(audit.read_text())["seeds"]
    private={x["task_id"]:x for x in rows(root/"frozen96/private96_tests.jsonl")}
    verify_image()
    results=[]
    for seed in (42,43,44):
        generation={(x["task_id"],x["sample_id"],x["arm"]):x for x in rows(root/f"eval_seed{seed}_96/checkpoint.jsonl")}
        for f in flips[str(seed)]["flips"]:
            if f["direction"]!="regression" or f["grpo_reason"]!="NameError":continue
            raw=generation[f["task_id"],f["sample_id"],"grpo"]["raw_code"]
            code,_=extract_code(raw)
            if (not re.search(r"\bList\s*\[",code)
                or re.search(r"from\s+typing\s+import\s+.*\bList\b",code)):
                continue
            if re.search(r"^\s*from\s+__future__\s+import",code,re.M):
                raise ValueError("future import found; prepend would change syntax")
            # One-line intervention; original source remains SHA-frozen.
            fixed="from typing import List\n"+code
            oracle=private[f["task_id"]]
            verdict=run_docker(fixed,oracle["test"],oracle["entry_point"])
            results.append({"seed":seed,"task_id":f["task_id"],"sample_id":f["sample_id"],
                "original_error":"NameError","counterfactual_passed":verdict["passed"],
                "counterfactual_error":verdict["reason"],
                "original_code_sha256":hashlib.sha256(code.encode()).hexdigest(),
                "intervention":"prepend_from_typing_import_List"})
    return {"experiment":"EXP-022","scope":"paired GRPO regressions with missing typing.List",
            "count":len(results),"recovered":sum(x["counterfactual_passed"] for x in results),
            "original_benchmark_scores_unchanged":True,
            "method_limit":"Retrospectively selected error group; explanatory intervention, not new GRPO performance estimate.",
            "results":results}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--audit",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=run(a.root,a.audit)
    if a.out.exists():raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"count":result["count"],"recovered":result["recovered"]}))
if __name__=="__main__":main()
