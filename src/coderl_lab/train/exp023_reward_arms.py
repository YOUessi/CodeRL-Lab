"""EXP-023 mutually exclusive reward ablation heads over frozen 7B GRPO.
format: static Python+solve validity, no hidden tests.
execution: Docker first-stage executable reward, ignores fraction correct.
full: original 0.1+0.9*fraction-passed reward.
"""
import json
from pathlib import Path
from coderl_lab.train.train014b_reward_sandbox import (validate_tests,normalize_code,check_code,score_one)
ARMS=("format","execution","full")
def ablation_score(completion,cases,arm):
    if arm not in ARMS:raise ValueError(arm)
    tests=validate_tests(cases)
    code,_=normalize_code(completion)
    valid,entry=check_code(code)
    if arm=="format":return {"reward":float(valid and entry),"python_valid":valid,"entry_valid":entry,"verdict":"static_format"}
    result=score_one(completion,tests)
    if arm=="full":return result
    # Execution proxy: Docker completed and candidate did not raise a top-level exception.
    # wrong_answer may mask per-case exceptions; this is NOT a pure execution oracle.
    return {**result,"reward":float(result["verdict"] in ("none","wrong_answer"))}
def reward_factory_ablated(logfile:Path,arm:str):
    if arm not in ARMS:raise ValueError(arm)
    logfile.parent.mkdir(parents=True,exist_ok=True)
    def reward_func(prompts,completions,completion_ids=None,task_id=None,test_cases=None,**kwargs):
        if task_id is None or test_cases is None or not(len(prompts)==len(completions)==len(task_id)==len(test_cases)):
            raise ValueError("invalid metadata for GRPO reward")
        rewards=[]
        for qid,completion,tests in zip(task_id,completions,test_cases):
            x=ablation_score(completion,tests,arm)
            with logfile.open("a",encoding="utf-8") as f:
                f.write(json.dumps({"task_id":qid,"reward_arm":arm,**x},ensure_ascii=False)+"\n")
            rewards.append(x["reward"])
        return rewards
    reward_func.__name__="exp023_"+arm+"_reward"
    return reward_func
