"""Sandbox code-correctness reward for TRAIN-014 GRPO.

Generated Python code is NEVER executed on the Tang host. Every candidate
runs inside an immutable pinned, unprivileged network-disabled Docker image.
"""
from __future__ import annotations
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

IMAGE="python:3.11-slim"
IMAGE_ID="sha256:d3846bce6e97d0ef5c32fc023300d5e39251ad791e6fdfaee41d2ba75246766e"
TIMEOUT=8
TICK=chr(96)*3
OPEN=re.compile(r"^"+re.escape(TICK)+r"(?:python|py)?[ \t]*\r?\n",re.I)
CLOSE=re.compile(r"\r?\n"+re.escape(TICK)+r"(?=[ \t]*(?:\r?\n|$))")

DRIVER=r'''
import json,sys,io,contextlib
p=json.load(sys.stdin)
space={"__name__":"__train014_candidate__"}
try:
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        exec(compile(p["code"],"<candidate>","exec"),space,space)
        solve=space["solve"]
        if not callable(solve):raise TypeError("solve not callable")
        n=0
        for t in p["tests"]:
            val=solve(t["x"])
            if type(val) is not int or val!=t["expected"]:
                outcome={"correct":False,"passed":n,"error":"wrong_answer"}
                break
            n+=1
        else:
            outcome={"correct":True,"passed":n,"error":"none"}
except BaseException as exc:
    outcome={"correct":False,"passed":0,"error":type(exc).__name__}
print("__TRAIN014__"+json.dumps(outcome))
'''

def text_sha(s:str)->str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def normalize_code(value:Any)->tuple[str,str]:
    if isinstance(value,list):
        if not value or not isinstance(value[-1],dict):
            raise ValueError("malformed conversational GRPO completion")
        value=value[-1].get("content","")
    if not isinstance(value,str):
        raise TypeError("completion must be text/messages")
    t=value.strip()
    start=OPEN.match(t)
    if not start:
        return t,"not_complete_leading_fence"
    end=CLOSE.search(t,start.end())
    if end is None:
        return t,"unclosed_fence"
    return t[start.end():end.start()].strip(),"first_closed_fence"

def check_code(code:str)->tuple[bool,bool]:
    if not code or len(code)>16000:
        return False,False
    try: parsed=ast.parse(code)
    except (SyntaxError,ValueError,RecursionError,MemoryError): return False,False
    return True,any(isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name=="solve" for x in parsed.body)

def validate_tests(cases:Any)->list[dict]:
    if not isinstance(cases,list) or not 3<=len(cases)<=20:
        raise ValueError("reward requires 3..20 integer tests")
    seen=set();result=[]
    for row in cases:
        if (not isinstance(row,dict) or set(row)!={"x","expected"}
            or type(row["x"]) is not int or type(row["expected"]) is not int
            or abs(row["x"])>10000 or abs(row["expected"])>10**9
            or row["x"] in seen):
            raise ValueError("malformed or duplicated reward test")
        seen.add(row["x"]);result.append(row)
    return result

def check_pinned_docker()->str:
    p=subprocess.run(["docker","image","inspect",IMAGE,"--format","{{.Id}}"],
                     text=True,capture_output=True,timeout=15,check=True)
    if p.stdout.strip()!=IMAGE_ID:
        raise RuntimeError("Docker image changed, stop evaluation")
    return IMAGE_ID

def docker_test(code:str,cases:list[dict])->dict:
    tests=validate_tests(cases)
    payload=json.dumps({"code":code,"tests":tests},ensure_ascii=False)
    cmd=[
        "docker","run","--rm","-i","--network","none",
        "--memory","256m","--cpus","1","--pids-limit","32",
        "--read-only","--cap-drop","ALL","--security-opt","no-new-privileges",
        "--user","65534:65534","--tmpfs","/tmp:rw,nosuid,size=16m",
        "--workdir","/tmp","-e","PYTHONDONTWRITEBYTECODE=1",
        "--entrypoint","python",IMAGE,"-I","-c",DRIVER,
    ]
    try:
        p=subprocess.run(cmd,input=payload,text=True,capture_output=True,timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"correct":False,"passed":0,"error":"candidate_timeout"}
    if p.returncode==137:
        return {"correct":False,"passed":0,"error":"candidate_memory_limit"}
    if p.returncode!=0:
        raise RuntimeError("Unknown Docker/checker failure rc="+str(p.returncode)+" stderr_tail="+p.stderr[-160:])
    marker="__TRAIN014__"
    lines=[s[len(marker):] for s in p.stdout.splitlines() if s.startswith(marker)]
    if len(lines)!=1:
        raise RuntimeError("sandbox returned no unique structured verdict")
    value=json.loads(lines[0])
    if (type(value.get("correct")) is not bool
        or type(value.get("passed")) is not int
        or not isinstance(value.get("error"),str)):
        raise RuntimeError("sandbox verdict malformed")
    if value["correct"] and value["passed"]!=len(tests):
        raise RuntimeError("sandbox verdict test count inconsistent")
    return value

def score_one(completion:Any,cases:Any)->dict:
    tests=validate_tests(cases)
    code,normalizer=normalize_code(completion)
    valid,entry=check_code(code)
    basic={"code_sha256":text_sha(code),"normalizer":normalizer,
           "python_valid":valid,"entry_valid":entry}
    if not valid or not entry:
        return {**basic,"reward":0.0,"all_passed":False,"verdict":"invalid_code"}
    res=docker_test(code,tests)
    return {**basic,"reward":1.0 if res["correct"] else 0.1,
            "all_passed":res["correct"],"verdict":res["error"]}

def reward_factory(logfile:Path):
    logfile.parent.mkdir(parents=True,exist_ok=True)
    def reward_func(prompts,completions,completion_ids=None,task_id=None,test_cases=None,**kwargs):
        if task_id is None or test_cases is None:
            raise ValueError("TRL GRPO task_id/test_cases metadata missing")
        if not (len(prompts)==len(completions)==len(task_id)==len(test_cases)):
            raise ValueError("TRL GRPO reward metadata/sample count mismatch")
        rewards=[]
        for qid,generated,tests in zip(task_id,completions,test_cases):
            stats=score_one(generated,tests)
            row={"task_id":qid,**stats}
            with logfile.open("a",encoding="utf-8") as f:
                f.write(json.dumps(row,ensure_ascii=False)+"\n")
                f.flush()
            rewards.append(stats["reward"])
        return rewards
    reward_func.__name__="sandbox_python_correctness_reward"
    return reward_func
