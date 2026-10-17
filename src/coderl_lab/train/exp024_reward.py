"""EXP-024: strict per-case execution, unresolved-name and semantic reward.
Runs all generated candidates ONLY inside the pinned unprivileged Docker image.
"""
from __future__ import annotations
import json,subprocess,hashlib
from pathlib import Path
from coderl_lab.train.train014b_reward_sandbox import (
    IMAGE,IMAGE_ID,TIMEOUT,normalize_code,check_code,validate_tests,check_pinned_docker
)
DRIVER=r'''
import json,sys,contextlib,io
p=json.load(sys.stdin)
s={"__name__":"__candidate__"}
out={"loaded":False,"load_error":None,"cases":[]}
try:
 with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
  exec(compile(p["code"],"<candidate>","exec"),s,s)
  fn=s["solve"]
  if not callable(fn):raise TypeError("solve not callable")
 out["loaded"]=True
except BaseException as ex:
 out["load_error"]=type(ex).__name__
if out["loaded"]:
 for t in p["tests"]:
  try:
   with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
    val=fn(t["x"])
   out["cases"].append({"executed":True,"correct":type(val) is int and val==t["expected"],"error":"none"})
  except BaseException as ex:
   out["cases"].append({"executed":False,"correct":False,"error":type(ex).__name__})
print("__EXP024__"+json.dumps(out))
'''
def docker_probe(code,tests):
    check_pinned_docker()
    cmd=["docker","run","--rm","-i","--network","none","--memory","256m",
         "--cpus","1","--pids-limit","32","--read-only","--cap-drop","ALL",
         "--security-opt","no-new-privileges","--user","65534:65534",
         "--tmpfs","/tmp:rw,nosuid,size=16m","--workdir","/tmp",
         "-e","PYTHONDONTWRITEBYTECODE=1","--entrypoint","python",
         IMAGE,"-I","-c",DRIVER]
    try:
        p=subprocess.run(cmd,input=json.dumps({"code":code,"tests":tests}),text=True,
                         capture_output=True,timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"loaded":False,"load_error":"timeout","cases":[]}
    if p.returncode==137:return {"loaded":False,"load_error":"memory_limit","cases":[]}
    if p.returncode!=0:raise RuntimeError("Docker infrastructure error "+str(p.returncode))
    lines=[s[len("__EXP024__"):] for s in p.stdout.splitlines() if s.startswith("__EXP024__")]
    if len(lines)!=1:raise RuntimeError("missing unique Docker probe verdict")
    ans=json.loads(lines[0])
    if not isinstance(ans.get("loaded"),bool) or not isinstance(ans.get("cases"),list):
        raise RuntimeError("malformed probe verdict")
    if ans["loaded"] and len(ans["cases"])!=len(tests):raise RuntimeError("incomplete cases")
    return ans
def score(completion,cases):
    tests=validate_tests(cases)
    code,norm=normalize_code(completion)
    valid,entry=check_code(code)
    base={"syntax_and_entry":bool(valid and entry),"normalizer":norm,
          "code_sha256":hashlib.sha256(code.encode()).hexdigest()}
    if not valid or not entry:
        return {**base,"reward":0.0,"executable_fraction":0.,
                "dependency_fraction":0.,"semantic_fraction":0.,"verdict":"invalid_code"}
    result=docker_probe(code,tests)
    if not result["loaded"]:
        return {**base,"reward":0.0,"executable_fraction":0.,
                "dependency_fraction":0.,"semantic_fraction":0.,
                "verdict":result["load_error"]}
    rows=result["cases"]
    executed=sum(x["executed"] for x in rows)/len(rows)
    dep=sum(x["error"]!="NameError" for x in rows)/len(rows)
    semantic=sum(x["correct"] for x in rows)/len(rows)
    # Performance signal dominated by semantic tests; no reward 1.0 for wrong answers.
    reward=round(.10*executed+.10*dep+.80*semantic,6)
    return {**base,"reward":reward,"executable_fraction":executed,
            "dependency_fraction":dep,"semantic_fraction":semantic,
            "verdict":"completed","load_error":None}
def reward_factory(logfile:Path):
    logfile.parent.mkdir(parents=True,exist_ok=True)
    def callback(prompts,completions,completion_ids=None,task_id=None,test_cases=None,**kwargs):
        if task_id is None or test_cases is None or not(len(prompts)==len(completions)==len(task_id)==len(test_cases)):
            raise ValueError("GRPO reward metadata missing/mismatched")
        values=[]
        for q,c,t in zip(task_id,completions,test_cases):
            outcome=score(c,t)
            with logfile.open("a",encoding="utf8") as f:
                f.write(json.dumps({"task_id":q,**outcome},ensure_ascii=False)+"\n")
            values.append(outcome["reward"])
        return values
    callback.__name__="exp024_strict_reward"
    return callback
