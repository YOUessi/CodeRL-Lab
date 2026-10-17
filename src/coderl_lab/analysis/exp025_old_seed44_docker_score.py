"""EXP-019B: pinned MBPP+ enhanced-test Docker scorer for frozen 7B codes.

This is NOT official EvalPlus leaderboard grading: runs this pinned Parquet
subset's embedded test script under a fixed 15-second secure Docker profile.
Never execute candidate code on the Tang host. Reference solution controls
are run only AFTER all 192 model completions are frozen. No cherry-picking.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import subprocess
from collections import Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed

from coderl_lab.datasets.exp019b_mbppplus_freeze import digest
from coderl_lab.analysis.exp025_old_seed44_generate import (
    ARMS,DRAWS,MODEL,REVISION,SFT_SHA,GRPO_SHA,read_jsonl,read_checkpoint,
    validate_public,
)

IMAGE="semaremask-evalplus:latest"
IMAGE_ID="sha256:8734cd8b7868a1b0205384477cfdef64753ff979eb45d0249f60dc31a64c74ea"
TIMEOUT_SEC=15
TOTAL=96*DRAWS*len(ARMS)

DRIVER=r"""
import contextlib, io, json,sys
p=json.load(sys.stdin)
scope={"__name__":"__mbppplus_candidate__"}
try:
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        exec(compile(p["code"],"<generated_candidate>","exec"),scope,scope)
        entry=scope[p["entry_point"]]
        if not callable(entry):
            raise TypeError("MBPP entry point is not callable")
        exec(compile(p["test"],"<pinned_mbppplus_test>","exec"),scope,scope)
except BaseException as ex:
    print("__EXP019B__"+json.dumps({"passed":False,"reason":type(ex).__name__}))
else:
    print("__EXP019B__"+json.dumps({"passed":True,"reason":"none"}))
"""


def text_sha(s:str)->str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def extract_code(raw:str)->tuple[str,str]:
    t=raw.strip()
    marker=chr(96)*3
    if t.startswith(marker):
        line_break=t.find("\n")
        if line_break<0:
            return t,"unclosed_leading_fence"
        lang=t[len(marker):line_break].strip().lower()
        if lang not in ("","py","python"):
            return t,"not_python_leading_fence"
        end=t.find("\n"+marker,line_break+1)
        if end<0:
            return t,"unclosed_leading_fence"
        return t[line_break+1:end].strip(),"first_closed_python_fence"
    return t,"unfenced_unmodified"


def ast_valid(code:str,entrypoint:str)->tuple[bool,bool]:
    if not code or len(code)>18000:
        return False,False
    try:parsed=ast.parse(code)
    except (SyntaxError,ValueError,RecursionError,MemoryError):
        return False,False
    entry=any(isinstance(item,ast.FunctionDef) and item.name==entrypoint
              for item in parsed.body)
    return True,entry


def verify_image()->str:
    p=subprocess.run(
        ["docker","image","inspect",IMAGE,"--format","{{.Id}}"],
        capture_output=True,text=True,timeout=20,check=True,
    )
    if p.stdout.strip()!=IMAGE_ID:
        raise RuntimeError("MBPP+ benchmark Docker image SHA mismatch")
    return IMAGE_ID


def run_docker(code:str,test:str,entrypoint:str)->dict:
    payload=json.dumps({"code":code,"test":test,"entry_point":entrypoint},ensure_ascii=False)
    cmd=[
        "docker","run","--rm","-i","--network","none",
        "--memory","1g","--cpus","1","--pids-limit","64",
        "--read-only","--cap-drop","ALL",
        "--security-opt","no-new-privileges",
        "--user","65534:65534",
        "--tmpfs","/tmp:rw,nosuid,size=128m",
        "-e","PYTHONDONTWRITEBYTECODE=1","-e","OPENBLAS_NUM_THREADS=1",
        "--entrypoint","/usr/bin/python3",
        IMAGE,"-c",DRIVER,
    ]
    try:
        proc=subprocess.run(cmd,input=payload,text=True,capture_output=True,
                            timeout=TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        return {"passed":False,"reason":"candidate_or_test_timeout"}
    if proc.returncode==137:
        return {"passed":False,"reason":"candidate_or_test_memory_limit"}
    if proc.returncode!=0:
        raise RuntimeError(
            "Unclassified MBPP+ Docker test environment failure "+
            f"rc={proc.returncode}; stderr_tail={proc.stderr[-180:]}"
        )
    label="__EXP019B__"
    lines=[s[len(label):] for s in proc.stdout.splitlines() if s.startswith(label)]
    if len(lines)!=1:
        raise RuntimeError("secure MBPP+ sandbox did not return one unique verdict")
    result=json.loads(lines[0])
    if type(result.get("passed")) is not bool or not isinstance(result.get("reason"),str):
        raise RuntimeError("secure Docker scorer returned malformed verdict")
    return result


def paired_ci(diffs:list[float],reps:int=20000)->dict:
    if len(diffs)!=96:raise ValueError("Bootstrap by exactly 96 tasks")
    rng=random.Random(42);n=len(diffs)
    v=sorted(sum(diffs[rng.randrange(n)] for _ in range(n))/n for _ in range(reps))
    return {
        "estimate":sum(diffs)/n,
        "ci95_low":v[int(.025*reps)],
        "ci95_high":v[int(.975*reps)-1],
        "iterations":reps,"seed":42,"resample_unit":"MBPPPlus_original_task",
    }


def summarize(public:list[dict],scored:dict,gold:dict,gen:dict)->dict:
    tasks=[]
    for row in public:
        q=row["task_id"]
        per={}
        for arm in ARMS:
            outcomes=[scored[(q,i,arm)] for i in range(DRAWS)]
            passbits=[int(out["passed"]) for out in outcomes]
            per[arm]={
                "candidate_correct":sum(passbits),
                "pass_at1":passbits[0],
                "observed_solved_at4":int(any(passbits)),
            }
        tasks.append({"task_id":q,"golden_test_passed":gold[q]["passed"],**per})
    stats={}
    for arm in ARMS:
        stats[arm]={
            "candidate_correct":sum(row[arm]["candidate_correct"] for row in tasks),
            "candidate_total":96*DRAWS,
            "candidate_correct_fraction":sum(row[arm]["candidate_correct"] for row in tasks)/(96*DRAWS),
            "pass_at1":sum(row[arm]["pass_at1"] for row in tasks),
            "pass_at1_total":96,
            "observed_solved_at4":sum(row[arm]["observed_solved_at4"] for row in tasks),
            "observed_solved_at4_total":96,
            "syntax_valid_count":sum(scored[(r["task_id"],i,arm)]["syntax_valid"]
                                      for r in public for i in range(DRAWS)),
            "top_level_entry_count":sum(scored[(r["task_id"],i,arm)]["top_level_entry"]
                                       for r in public for i in range(DRAWS)),
            "cap256_hit_count":sum(gen[(r["task_id"],i,arm)]["generated_tokens"]>=256
                                    for r in public for i in range(DRAWS)),
            "verdict_types":dict(Counter(scored[(r["task_id"],i,arm)]["reason"]
                                      for r in public for i in range(DRAWS))),
        }
    result={
        "task_count":96,"candidate_total":TOTAL,
        "scores_by_arm":stats,
        "reference_solution_test_status":{
            "passed":sum(gold[q]["passed"] for q in gold),
            "failed":sum(not gold[q]["passed"] for q in gold),
            "failure_types":dict(Counter(gold[q]["reason"] for q in gold if not gold[q]["passed"])),
        },
        "paired_task_bootstrap":{
            "candidate_correct_fraction":paired_ci([
                (x["grpo"]["candidate_correct"]-x["sft"]["candidate_correct"])/DRAWS
                for x in tasks]),
            "pass_at1":paired_ci([
                x["grpo"]["pass_at1"]-x["sft"]["pass_at1"] for x in tasks]),
            "observed_solved_at4":paired_ci([
                x["grpo"]["observed_solved_at4"]-x["sft"]["observed_solved_at4"]
                for x in tasks]),
        },
    }
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--public",type=Path,required=True)
    p.add_argument("--private",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--generation-dir",type=Path,required=True)
    p.add_argument("--workers",type=int,default=2)
    a=p.parse_args()
    if a.workers not in (1,2):
        raise ValueError("MBPP+ 1GB Docker workers must be 1..2")
    public,manifest=validate_public(a.public,a.manifest)
    freeze=json.loads((a.generation_dir/"generation_summary.json").read_text())
    identity=json.loads((a.generation_dir/"identity.json").read_text())
    checkpoint=a.generation_dir/"checkpoint.jsonl"
    # Complete and inspect SHA-frozen GENERATED CODES BEFORE touching hidden tests.
    if (freeze.get("status")!="all_samples_generated_frozen"
        or freeze.get("answers_generated")!=TOTAL
        or freeze.get("checkpoint_sha256")!=digest(checkpoint)
        or freeze.get("private_mbppplus_test_labels_opened") is not False
        or freeze.get("num_tasks")!=96
        or identity.get("expected_output_count")!=TOTAL
        or identity.get("private_oracle_sha256_only")!=manifest["private_sha256"]
        or freeze.get("model_adapters_sha256")!={"sft":SFT_SHA,"grpo":GRPO_SHA}):
        raise ValueError("full 192 code outputs not SHA-frozen BEFORE hidden test access")
    generated=read_checkpoint(checkpoint,public)
    expected={(r["task_id"],i,arm) for r in public for i in range(DRAWS) for arm in ARMS}
    if set(generated)!=expected:
        raise ValueError("SFT and GRPO generated codes not paired/frozen for all96 tasks")

    # First access to hidden test bytes occurs only after the validation above.
    if digest(a.private)!=manifest["private_sha256"]:
        raise ValueError("MBPP+ test oracle SHA changed")
    private=read_jsonl(a.private)
    if len(private)!=96 or [p["task_id"] for p in private]!=[p["task_id"] for p in public]:
        raise ValueError("oracle/public task ID mismatch")
    oracle={p["task_id"]:p for p in private}
    for r in public:
        if oracle[r["task_id"]]["entry_point"]!=r["entry_point"]:
            raise ValueError("benchmark expected function entry drift")
    verify_image()
    original_code_sha=digest(Path(__file__))
    score_identity={
        "experiment":"EXP-019B-PINNED-DOCKER-ENHANCED-MBPP",
        "source_public_sha256":digest(a.public),
        "source_private_sha256":digest(a.private),
        "frozen_generation_checkpoint_sha256":digest(checkpoint),
        "generation_summary_sha256":digest(a.generation_dir/"generation_summary.json"),
        "scorer_source_sha256":original_code_sha,
        "image_id":IMAGE_ID,"timeout_seconds":TIMEOUT_SEC,
        "docker":"no_net/nonroot/read_only/no_caps/1GB/1CPU",
        "scope":"96 test-field scripts, not native EvalPlus leaderboard timing or Base/Extra score",
        "models":{"sft":SFT_SHA,"grpo":GRPO_SHA},
    }
    marker=a.generation_dir/"score_identity.json"
    if marker.exists():
        if json.loads(marker.read_text())!=score_identity:
            raise ValueError("scorer identity changed during resume")
    else:marker.write_text(json.dumps(score_identity,indent=2)+"\n")

    # Record golden-reference test execution on the same pinned container,
    # but do not use it to choose/drop tasks or tune evaluation parameters.
    goldfile=a.generation_dir/"golden_control_checkpoint.jsonl"
    gold={}
    if goldfile.exists():
        for rec in read_jsonl(goldfile):
            if rec["task_id"] in gold:
                raise ValueError("golden control duplicate")
            if rec["source_sha256"]!=oracle[rec["task_id"]]["source_sha256"]:
                raise ValueError("golden control source drift")
            gold[rec["task_id"]]=rec
    for q in [p["task_id"] for p in public]:
        if q in gold:continue
        row=oracle[q]
        verdict=run_docker(row["canonical_code"],row["test"],row["entry_point"])
        rec={"task_id":q,"source_sha256":row["source_sha256"],**verdict}
        with goldfile.open("a",encoding="utf-8") as f:
            f.write(json.dumps(rec)+"\n");f.flush()
        gold[q]=rec
        print(f"MBPP+ reference environment controls {len(gold)}/96",flush=True)

    scorepath=a.generation_dir/"docker_score_checkpoint.jsonl"
    scored={}
    if scorepath.exists():
        for rec in read_jsonl(scorepath):
            key=(rec["task_id"],int(rec["sample_id"]),rec["arm"])
            if key in scored or key not in generated or rec["code_sha256"]!=generated[key]["code_sha256"]:
                raise ValueError("candidate scoring checkpoint ID/content mismatch")
            scored[key]=rec

    def evaluate(key):
        gen=generated[key]
        row=oracle[key[0]]
        code,normalizer=extract_code(gen["raw_code"])
        syntax,entry=ast_valid(code,row["entry_point"])
        if not syntax or not entry:
            outcome={"passed":False,"reason":"invalid_syntax_or_entry"}
        else:
            outcome=run_docker(code,row["test"],row["entry_point"])
        return key,{
            "task_id":key[0],"sample_id":key[1],"arm":key[2],
            "code_sha256":gen["code_sha256"],
            "normalized_code_sha256":text_sha(code),
            "normalizer":normalizer,
            "syntax_valid":syntax,"top_level_entry":entry,
            **outcome,
        }
    remaining=[key for key in sorted(expected) if key not in scored]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        active={pool.submit(evaluate,key):key for key in remaining}
        for fut in as_completed(active):
            key,record=fut.result()  # infrastructure failures stop run
            with scorepath.open("a",encoding="utf-8") as f:
                f.write(json.dumps(record,ensure_ascii=False)+"\n")
                f.flush()
            scored[key]=record
            if len(scored)%16==0:
                print(f"MBPP+ model candidate Docker verdicts {len(scored)}/{TOTAL}",flush=True)
    if set(scored)!=expected:
        raise ValueError("not complete full768 fixed candidates")
    summary={
        **score_identity,
        "status":"completed_pinned_MBPPPlus_96_tasks",
        **summarize(public,scored,gold,generated),
        "golden_control_checkpoint_sha256":digest(goldfile),
        "candidate_scoring_checkpoint_sha256":digest(scorepath),
        "test_labels_accessed_only_after_generation_frozen":True,
        "raw_generated_code_and_reference_not_uploaded":True,
        "limitations":"Historical public MBPP+ test subset; image lacks some optional Python packages; fixed15s timeout not calibrated to golden runtime like native EvalPlus. Golden failures are disclosed and NEVER silently filtered from 24-task denominator.",
    }
    out=a.generation_dir/"score_summary.json"
    if out.exists():
        if json.loads(out.read_text())!=summary:
            raise FileExistsError("benchmark score report mismatch; refusing overwrite")
    else:out.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
