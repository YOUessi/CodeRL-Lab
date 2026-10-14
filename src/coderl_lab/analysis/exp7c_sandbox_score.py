"""EXP-7C historical HumanEval+ pinned-assertion evaluator in isolated Docker.

Not the native EvalPlus CLI Base/Extra split. Original 1312 completions are
frozen before this process reads any tests; no generated code runs on the host.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from coderl_lab.datasets.exp7c_humanevalplus import SOURCE_SHA256, sha256_file
from coderl_lab.analysis.exp7c_matched_generate import (
    ARMS, ADAPTER_SHA, CONFIG, read_public, read_checkpoint,
)

IMAGE = "semaremask-evalplus:latest"
IMAGE_ID = "sha256:8734cd8b7868a1b0205384477cfdef64753ff979eb45d0249f60dc31a64c74ea"
TIMEOUT_SECONDS = 12

DRIVER = r"""
import contextlib, io, json, sys
data=json.load(sys.stdin)
scope={"__name__":"__main__"}
try:
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        exec(compile(data["code"],"<candidate>","exec"),scope,scope)
        fn=scope[data["entry_point"]]
        if not callable(fn): raise TypeError("entrypoint_not_callable")
        exec(compile(data["test"],"<official_test>","exec"),scope,scope)
        scope["check"](fn)
except BaseException as exc:
    print("__CODERL_EXP7C__"+json.dumps({"passed":False,"kind":type(exc).__name__}))
else:
    print("__CODERL_EXP7C__"+json.dumps({"passed":True,"kind":"none"}))
"""


def image_sha() -> str:
    p=subprocess.run(
        ["docker","image","inspect",IMAGE,"--format","{{.Id}}"],
        capture_output=True,text=True,check=True,
    )
    value=p.stdout.strip()
    if value!=IMAGE_ID:
        raise ValueError("pinned Docker image changed: "+value)
    return value


def sandbox_check(code: str, test: str, entrypoint: str) -> tuple[bool, str]:
    args=[
        "docker","run","--rm","-i","--network","none",
        "--memory","1g","--cpus","1","--pids-limit","64",
        "--read-only","--cap-drop","ALL","--security-opt","no-new-privileges",
        "--user","65534:65534","--tmpfs","/tmp:rw,nosuid,size=128m",
        "-e","PYTHONDONTWRITEBYTECODE=1","-e","OPENBLAS_NUM_THREADS=1",
        "--entrypoint","/usr/bin/python3",IMAGE,"-c",DRIVER,
    ]
    try:
        p=subprocess.run(
            args,input=json.dumps({"code":code,"test":test,"entry_point":entrypoint}),
            text=True,capture_output=True,timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return False,"candidate_or_checker_timeout"
    if p.returncode==137:
        return False,"candidate_memory_limit"
    if p.returncode!=0:
        if (
            "Fatal Python error: Segmentation fault" in p.stderr
            and 'File "<candidate>"' in p.stderr
        ):
            return False,"candidate_native_segfault"
        raise RuntimeError(
            f"unclassified Docker/evaluator failure rc={p.returncode}: {p.stderr[-300:]}"
        )
    mark="__CODERL_EXP7C__"
    outputs=[line[len(mark):] for line in p.stdout.splitlines() if line.startswith(mark)]
    if len(outputs)!=1:
        raise RuntimeError("isolated evaluator returned no unique structured verdict")
    x=json.loads(outputs[0])
    if not isinstance(x.get("passed"),bool):
        raise RuntimeError("invalid verdict type")
    return x["passed"],str(x.get("kind","unknown"))


def bootstrap_deltas(rows: list[dict], iterations: int=20000, seed: int=42) -> dict:
    effects=[int(r["sft"])-int(r["base"]) for r in rows]
    n=len(effects)
    rng=random.Random(seed)
    values=sorted(
        sum(effects[rng.randrange(n)] for _ in range(n))/n
        for _ in range(iterations)
    )
    return {
        "observed_delta":sum(effects)/n,
        "ci95_low":values[int(iterations*0.025)],
        "ci95_high":values[int(iterations*0.975)-1],
        "iterations":iterations,"seed":seed,
    }


def summarize(rows: dict) -> dict:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda:{arm:[] for arm in ARMS})
    for (task_id,arm,n),result in rows.items():
        grouped[task_id][arm].append(result)
    tasks=[]
    for qid,arms in sorted(grouped.items(),key=lambda x:int(x[0].split("/")[1])):
        if any(len(arms[arm])!=4 for arm in ARMS):
            raise RuntimeError("incomplete same-budget samples")
        tasks.append({
            "task_id":qid,
            "base":any(x["passed"] for x in arms["base"]),
            "sft":any(x["passed"] for x in arms["sft"]),
        })
    if len(tasks)!=164:
        raise RuntimeError("full 164 tasks required")
    per_arm={
        arm:{
            "sample_correct":sum(bool(x["passed"]) for (qid,a,s),x in rows.items() if a==arm),
            "solved_at4":sum(int(x[arm]) for x in tasks),
            "failure_types":dict(Counter(
                x["reason"] for (qid,a,s),x in rows.items()
                if a==arm and not x["passed"]
            )),
        } for arm in ARMS
    }
    transitions={
        "wrong_to_correct":sum(not t["base"] and t["sft"] for t in tasks),
        "correct_to_wrong":sum(t["base"] and not t["sft"] for t in tasks),
        "correct_to_correct":sum(t["base"] and t["sft"] for t in tasks),
        "wrong_to_wrong":sum(not t["base"] and not t["sft"] for t in tasks),
    }
    return {
        "tasks":164,"samples":1312,"conditions":per_arm,
        "paired_transitions":transitions,
        "sft_minus_base_solved_at4":bootstrap_deltas(tasks),
    }


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--generation-dir",type=Path,required=True)
    p.add_argument("--public-tasks",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--source-jsonl",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args()
    public,manifest=read_public(a.public_tasks,a.manifest)
    generation=json.loads((a.generation_dir/"summary.json").read_text())
    checkpoint=a.generation_dir/"checkpoint.jsonl"
    checkpoint_sha=sha256_file(checkpoint)
    if any((
        generation.get("status")!="completed",
        generation.get("selection")!="full164",
        generation.get("sample_count")!=1312,
        generation.get("samples_per_task")!=4,
        generation.get("checkpoint_sha256")!=checkpoint_sha,
        generation.get("oracle_test_accessed_during_generation") is not False,
        generation.get("settings")!=CONFIG,
        generation.get("adapter_sha256")!=ADAPTER_SHA,
    )):
        raise ValueError("the full model generation is not frozen")
    expected={(r["task_id"],arm,n) for r in public for arm in ARMS for n in range(4)}
    candidates=read_checkpoint(checkpoint,public,4)
    if set(candidates)!=expected:
        raise ValueError("missing generated candidate pair")
    if sha256_file(a.source_jsonl)!=SOURCE_SHA256:
        raise ValueError("pinned HumanEval+ source SHA mismatch")
    image_sha()
    identity={
        "experiment":"EXP-7C",
        "source_sha256":SOURCE_SHA256,
        "public_sha256":manifest["public_sha256"],
        "generation_checkpoint_sha256":checkpoint_sha,
        "evaluator_source_sha256":sha256_file(Path(__file__)),
        "image_id":IMAGE_ID,
        "timeout_seconds":TIMEOUT_SECONDS,
        "memory":"1g","cpu":"1",
        "total_candidates":1312,"tasks":164,
        "scoring":"Pinned HumanEval+ test field, not native EvalPlus CLI Base/Extra",
    }
    a.output_dir.mkdir(parents=True,exist_ok=True)
    frozen=a.output_dir/"frozen_inputs.json"
    if frozen.exists():
        if json.loads(frozen.read_text())!=identity:
            raise ValueError("resuming with different evaluator inputs or code")
    else:
        frozen.write_text(json.dumps(identity,indent=2)+"\n")
    # First oracle test access takes place only AFTER model code/SHA freeze.
    source={}
    for line in a.source_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r=json.loads(line)
        qid=str(r["task_id"])
        if qid in source:
            raise ValueError("duplicate HumanEval+ original task ID")
        source[qid]={"test":str(r["test"]),"entry_point":str(r["entry_point"])}
    if set(source)!={r["task_id"] for r in public}:
        raise ValueError("source/public task ID mismatch")
    if any(r["entry_point"]!=source[r["task_id"]]["entry_point"] for r in public):
        raise ValueError("source/public entry points differ")

    scorepath=a.output_dir/"checkpoint.jsonl"
    scored={}
    if scorepath.exists():
        for line in scorepath.read_text().splitlines():
            if not line.strip(): continue
            row=json.loads(line)
            key=row["task_id"],row["arm"],int(row["sample_id"])
            if key in scored or key not in candidates:
                raise ValueError("duplicate/unrecognized scoring checkpoint")
            if row["code_sha256"]!=candidates[key]["code_sha256"]:
                raise ValueError("score checkpoint code SHA mismatch")
            scored[key]=row

    ordered=sorted(expected,key=lambda k:(int(k[0].split("/")[1]),k[2],ARMS.index(k[1])))
    for i,key in enumerate(ordered,1):
        if key in scored:
            continue
        c=candidates[key]
        if not c["python_ast_valid"]:
            passed,reason=False,"syntax_error"
        elif not c["entry_point_defined"]:
            passed,reason=False,"missing_entrypoint"
        else:
            passed,reason=sandbox_check(
                c["code"],source[key[0]]["test"],source[key[0]]["entry_point"]
            )
        row={
            "task_id":key[0],"arm":key[1],"sample_id":key[2],
            "code_sha256":c["code_sha256"],
            "passed":passed,"reason":"none" if passed else reason,
        }
        with scorepath.open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(row,ensure_ascii=False)+"\n")
            stream.flush()
        scored[key]=row
        if i%40==0:
            print(f"EXP-7C sandbox correct checks: {len(scored)}/1312",flush=True)
    if set(scored)!=expected:
        raise ValueError("incomplete official assertions")
    report={
        "experiment":"EXP-7C","status":"completed",
        "scope":"Pinned HumanEval+ generated test assertions; not native EvalPlus CLI Base/Extra split",
        "historical_benchmark_not_forward_temporal":True,
        "matched_model_base":"Qwen2.5-7B-Instruct",
        "adapter_sha256":ADAPTER_SHA,
        "frozen_inputs":identity,
        "checkpoint_sha256":sha256_file(scorepath),
        **summarize(scored),
    }
    output=a.output_dir/"summary.json"
    if output.exists():
        if json.loads(output.read_text())!=report:
            raise ValueError("refusing final score overwrite")
    else:
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
