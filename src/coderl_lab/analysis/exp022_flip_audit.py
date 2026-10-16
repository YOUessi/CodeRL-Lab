"""EXP-022: read-only paired mechanism audit of EXP-021 frozen 96-task results.

No candidate execution and no access to hidden oracle. Class labels are
heuristics for prioritizing manual review, not causal findings.
"""
from __future__ import annotations
import argparse
import ast
import collections
import difflib
import hashlib
import json
from pathlib import Path

SEEDS=(42,43,44)
DRAWS=4
EXPECTED_TASKS=96
def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()
def indexed(rows):
    d={}
    for row in rows:
        key=(row["task_id"],int(row["sample_id"]),row["arm"])
        if key in d:raise ValueError(f"duplicate {key}")
        d[key]=row
    return d
def parse_features(code):
    try:
        a=ast.parse(code)
    except (SyntaxError,ValueError,RecursionError):
        return {"ast_valid":False,"control_flow":None,"calls":None,"ast_dump_hash":None}
    nodes=list(ast.walk(a))
    structural=ast.dump(a,include_attributes=False)
    return {"ast_valid":True,
            "control_flow":sum(isinstance(n,(ast.If,ast.For,ast.While,ast.Try,ast.IfExp,ast.Match)) for n in nodes),
            "calls":sum(isinstance(n,ast.Call) for n in nodes),
            "ast_dump_hash":hashlib.sha256(structural.encode()).hexdigest()}
def audit(root:Path):
    report={"experiment":"EXP-022","source":"EXP-021-frozen96","seeds":{},"task_count":96,
            "method":"paired sample-id/read-only Docker verdict audit; AST structural heuristics",
            "no_hidden_tests_opened":True,"no_candidate_execution":True}
    all_records=[]
    for seed in SEEDS:
        d=root/f"eval_seed{seed}_96"
        gen_summary=json.loads((d/"generation_summary.json").read_text())
        score_summary=json.loads((d/"score_summary.json").read_text())
        generation_path=d/"checkpoint.jsonl"
        scoring_path=d/"docker_score_checkpoint.jsonl"
        if gen_summary.get("status")!="all_samples_generated_frozen" or gen_summary["checkpoint_sha256"]!=digest(generation_path):
            raise ValueError(f"seed{seed} generation not frozen")
        if score_summary.get("status")!="completed_pinned_MBPPPlus_96_tasks" or score_summary["candidate_scoring_checkpoint_sha256"]!=digest(scoring_path):
            raise ValueError(f"seed{seed} scoring not frozen")
        generations=indexed(read(generation_path))
        scored=indexed(read(scoring_path))
        task_ids={k[0] for k in generations}
        expected={(task,i,arm) for task in task_ids for i in range(DRAWS) for arm in ("sft","grpo")}
        if len(task_ids)!=EXPECTED_TASKS or set(generations)!=expected or set(scored)!=expected:
            raise ValueError(f"seed{seed} incomplete paired cohort")
        count=collections.Counter()
        categories=collections.Counter()
        disagreements=[]
        for task in sorted(task_ids):
            for sample in range(DRAWS):
                ga,gb=(generations[(task,sample,arm)] for arm in ("sft","grpo"))
                va,vb=(scored[(task,sample,arm)] for arm in ("sft","grpo"))
                if ga["code_sha256"]!=hashlib.sha256(ga["raw_code"].encode()).hexdigest() or gb["code_sha256"]!=hashlib.sha256(gb["raw_code"].encode()).hexdigest():
                    raise ValueError("checkpoint generated answer hash mismatch")
                if va["code_sha256"]!=ga["code_sha256"] or vb["code_sha256"]!=gb["code_sha256"]:
                    raise ValueError("generation-scoring mismatch")
                changed=ga["code_sha256"]!=gb["code_sha256"]
                outcome="repair" if not va["passed"] and vb["passed"] else "regression" if va["passed"] and not vb["passed"] else "unchanged_correct" if va["passed"] else "unchanged_wrong"
                count[outcome]+=1
                count["text_changed"]+=changed
                if not changed:continue
                fa,fb=parse_features(ga["raw_code"]),parse_features(gb["raw_code"])
                exact_ast_same=fa["ast_valid"] and fb["ast_valid"] and fa["ast_dump_hash"]==fb["ast_dump_hash"]
                heuristics={"ast_unchanged":bool(exact_ast_same),
                            "control_flow_changed":fa["control_flow"]!=fb["control_flow"],
                            "call_count_changed":fa["calls"]!=fb["calls"],
                            "text_similarity":round(difflib.SequenceMatcher(None,ga["raw_code"],gb["raw_code"],autojunk=False).ratio(),4)}
                categories[(outcome,va["reason"],vb["reason"])]+=1
                if outcome in ("repair","regression"):
                    disagreements.append({"task_id":task,"sample_id":sample,"direction":outcome,
                        "sft_reason":va["reason"],"grpo_reason":vb["reason"],"sft_code_sha256":ga["code_sha256"],
                        "grpo_code_sha256":gb["code_sha256"],**heuristics})
                all_records.append({"seed":seed,"task_id":task,"sample_id":sample,"outcome":outcome,
                                    "reasons":[va["reason"],vb["reason"]],**heuristics})
        report["seeds"][str(seed)]={"counts":dict(count),"transition_reasons":[{"direction":k[0],"before":k[1],"after":k[2],"count":v} for k,v in sorted(categories.items())],
                                   "flips":disagreements,"source_sha256":{"generation":digest(generation_path),"scoring":digest(scoring_path)}}
    # No raw model code, prompt or hidden oracle exported.
    report["all_changed_code_pair_count"]=len(all_records)
    report["limitations"]="AST and lexical heuristics do not prove semantic equivalence or reward causality; observed shifts depend on frozen sampling and test environment."
    return report
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=audit(a.root)
    if a.output.exists():raise FileExistsError("audit output exists; refusing overwrite")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"file":str(a.output),"changed":result["all_changed_code_pair_count"],"counts":{k:v["counts"] for k,v in result["seeds"].items()}},ensure_ascii=False))
if __name__=="__main__":main()
