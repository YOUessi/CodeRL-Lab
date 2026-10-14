import json
import shutil
import subprocess
import pytest
from coderl_lab.train.train014_reward_sandbox import (
    IMAGE_ID,check_code,normalize_code,validate_tests,
    check_pinned_docker,docker_test,score_one,reward_factory,
)
TESTS=[{"x":x,"expected":x+4} for x in [-5,-1,0,2,8]]

def test_frozen_code_extraction_and_no_python_import():
    tick=chr(96)*3
    candidate=tick+"python\ndef solve(x):\n    return x+4\n"+tick+"\nExplanation"
    code,reason=normalize_code(candidate)
    assert code=="def solve(x):\n    return x+4"
    assert reason=="first_closed_fence"
    assert check_code(code)==(True,True)
    assert check_code("print('hello')")==(True,False)
    assert score_one("not a program",TESTS)["reward"]==0.

def test_invalid_tests_fail_closed():
    with pytest.raises(ValueError):
        validate_tests([{"x":1,"expected":1}])
    with pytest.raises(ValueError):
        validate_tests([{"x":1,"expected":1}]*5)
    with pytest.raises(ValueError):
        validate_tests([{"x":1,"expected":True} for _ in range(5)])

def test_grpo_reward_receives_original_tests_and_hashes(tmp_path,monkeypatch):
    import coderl_lab.train.train014_reward_sandbox as src
    monkeypatch.setattr(src,"docker_test",lambda code,cases:{"correct":True,"passed":len(cases),"error":"none"})
    func=reward_factory(tmp_path/"rewards.jsonl")
    answer="def solve(x):\n    return x+4"
    result=func(
        prompts=[[{"role":"user","content":"function"}]],
        completions=[[{"role":"assistant","content":answer}]],
        task_id=["train/add/4"],test_cases=[TESTS],
    )
    assert result==[1.0]
    records=[json.loads(x) for x in (tmp_path/"rewards.jsonl").read_text().splitlines()]
    assert len(records)==1 and records[0]["reward"]==1.0
    assert "code" not in records[0] and "answer" not in records[0]

def pinned_image_available():
    if shutil.which("docker") is None:return False
    try:
        p=subprocess.run(["docker","image","inspect","python:3.11-slim","--format","{{.Id}}"],
                         text=True,capture_output=True,timeout=4)
    except (OSError,subprocess.TimeoutExpired):return False
    return p.returncode==0 and p.stdout.strip()==IMAGE_ID

@pytest.mark.skipif(not pinned_image_available(),reason="Requires pinned Docker image on Tang")
def test_real_docker_checker_passing_failing_and_syntax():
    assert check_pinned_docker()==IMAGE_ID
    good=score_one("def solve(x):\n    return x+4",TESTS)
    wrong=score_one("def solve(x):\n    return x+3",TESTS)
    invalid=score_one(chr(96)*3+"python\ndef solve(x): broken",TESTS)
    assert good["reward"]==1.0 and good["all_passed"]
    assert wrong["reward"]==0.1 and not wrong["all_passed"]
    assert invalid["reward"]==0.
