"""EXP-019: pinned Docker MBPP+ oracle, both advertised base and hidden plus.

No model-produced code is compiled or executed on the Tang host.
Candidate and test code run only inside a read-only, network-disabled
low-privilege Docker container with an immutable EvalPlus image ID.
This is NOT the native EvalPlus CLI dynamic-timeout benchmark scorer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

IMAGE="semaremask-evalplus:latest"
IMAGE_ID="sha256:8734cd8b7868a1b0205384477cfdef64753ff979eb45d0249f60dc31a64c74ea"
TIMEOUT_SECONDS=30

DRIVER=r'''
import sys,json,io,contextlib
payload=json.load(sys.stdin)
scope={"__name__":"__exp019_code_candidate__"}
try:
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        exec(compile(payload["code"],"<candidate>","exec"),scope,scope)
        for instruction in payload["base_test_list"]:
            exec(compile(instruction,"<base_public_example>","exec"),scope,scope)
        exec(compile(payload["augmented_test"],"<MBPPPlus_hidden_augmented>","exec"),scope,scope)
except BaseException as err:
    outcome={"passed":False,"error":type(err).__name__}
else:
    outcome={"passed":True,"error":"none"}
print("__CODERL_EXP019__"+json.dumps(outcome))
'''


def digest(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):
            h.update(b)
    return h.hexdigest()


def image_sha()->str:
    p=subprocess.run(
        ["docker","image","inspect",IMAGE,"--format","{{.Id}}"],
        text=True,capture_output=True,timeout=15,check=True,
    )
    actual=p.stdout.strip()
    if actual!=IMAGE_ID:
        raise ValueError("pinned EvalPlus Docker image identity changed: "+actual)
    return actual


def test_program(code:str,oracle:dict)->dict:
    if (not isinstance(code,str) or not code.strip() or len(code)>30000
        or not isinstance(oracle,dict)
        or set(oracle)!={"task_id","augmented_test","public_base_test_list","test_imports","test_sha256"}
        or oracle["test_imports"]!=[]
        or not isinstance(oracle["public_base_test_list"],list)
        or not 3<=len(oracle["public_base_test_list"])<=7
        or not isinstance(oracle["augmented_test"],str)
        or hashlib.sha256(oracle["augmented_test"].encode()).hexdigest()!=oracle["test_sha256"]):
        raise ValueError("missing or corrupted MBPP+ full hidden oracle/test input")
    payload={
        "code":code,"augmented_test":oracle["augmented_test"],
        "base_test_list":oracle["public_base_test_list"],
    }
    cmd=[
        "docker","run","--rm","-i","--network","none",
        "--memory","1g","--cpus","1","--pids-limit","64",
        "--read-only","--cap-drop","ALL","--security-opt","no-new-privileges",
        "--user","65534:65534","--tmpfs","/tmp:rw,nosuid,size=64m",
        "-e","PYTHONDONTWRITEBYTECODE=1","-e","OPENBLAS_NUM_THREADS=1",
        "-e","OMP_NUM_THREADS=1","--workdir","/tmp",
        "--entrypoint","/usr/bin/python3",IMAGE,"-I","-c",DRIVER,
    ]
    try:
        result=subprocess.run(
            cmd,input=json.dumps(payload,ensure_ascii=False),
            text=True,capture_output=True,timeout=TIMEOUT_SECONDS,check=False,
        )
    except subprocess.TimeoutExpired:
        return {"passed":False,"error":"candidate_or_checker_timeout"}
    if result.returncode==137:
        return {"passed":False,"error":"memory_limit"}
    if result.returncode!=0:
        raise RuntimeError(
            f"Unknown Docker/native judge error exit={result.returncode}: {result.stderr[-240:]}"
        )
    marker="__CODERL_EXP019__"
    replies=[s[len(marker):] for s in result.stdout.splitlines() if s.startswith(marker)]
    if len(replies)!=1:
        raise RuntimeError("Docker checker missing unique structured correctness verdict")
    result=json.loads(replies[0])
    if not isinstance(result.get("passed"),bool) or not isinstance(result.get("error"),str):
        raise RuntimeError("Docker checker verdict invalid")
    return result


def reference_preflight(oracle_path:Path,reference_path:Path,out:Path)->dict:
    image_sha()
    hidden=[json.loads(line) for line in oracle_path.read_text().splitlines() if line.strip()]
    refs=[json.loads(line) for line in reference_path.read_text().splitlines() if line.strip()]
    if (len(hidden)!=24 or len(refs)!=24
        or [r["task_id"] for r in hidden]!=[r["task_id"] for r in refs]):
        raise ValueError("frozen 24 reference/oracle tasks do not align")
    expected={
        "experiment":"EXP019-REFERENCES-IN-PINNED-DOCKER",
        "oracle_sha256":digest(oracle_path),
        "reference_sha256":digest(reference_path),
        "docker_image_sha256":IMAGE_ID,"checker_script_sha256":digest(Path(__file__)),
    }
    out.mkdir(parents=True,exist_ok=True)
    identity=out/"identity.json"
    if identity.exists():
        if json.loads(identity.read_text())!=expected:
            raise ValueError("reference benchmark check cannot reuse mutated inputs")
    else:
        identity.write_text(json.dumps(expected,indent=2)+"\n")
    path=out/"reference_checkpoint.jsonl"
    done={}
    if path.exists():
        for row in [json.loads(s) for s in path.read_text().splitlines() if s.strip()]:
            if row["task_id"] in done:raise ValueError("duplicate reference check")
            done[row["task_id"]]=row
    for i,(r,test) in enumerate(zip(refs,hidden),1):
        if r["task_id"] in done:continue
        source=r["reference_code"]
        if hashlib.sha256(source.encode()).hexdigest()!=r["reference_sha256"]:
            raise ValueError("official model-independent reference code changed")
        verdict=test_program(source,test)
        line={"task_id":r["task_id"],"passed":verdict["passed"],"error":verdict["error"],
              "reference_sha256":r["reference_sha256"]}
        with path.open("a") as f:
            f.write(json.dumps(line)+"\n");f.flush()
        done[r["task_id"]]=line
        if i%4==0:
            print(f"EXP019 official reference Docker checks: {len(done)}/24",flush=True)
    report={**expected,
            "status":"completed",
            "tasks_tested":len(done),
            "reference_pass":sum(r["passed"] for r in done.values()),
            "failure_by_type":{e:sum((not r["passed"]) and r["error"]==e for r in done.values())
                               for e in sorted({r["error"] for r in done.values()}) if e!="none"},
            "checkpoint_sha256":digest(path),
            "benchmark_reference_oracle_checked_before_any_model_generation":True,
            "no_candidate_model_outputs_scored":True}
    (out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--oracle",type=Path,required=True)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    reference_preflight(args.oracle,args.reference,args.output_dir)
