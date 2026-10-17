"""EXP-024 CPU unit tests: deterministic Docker verdict fixtures.

These tests deliberately DO NOT run containers on GitHub Actions.
The pinned-image Docker integration is tested separately on Tang.
"""
from coderl_lab.train import exp024_reward as reward

TESTS=[{"x":str(i),"expected":i} for i in range(10)]

def fake_probe(code,tests):
    if "x: List[int]" in code:
        return {"loaded":False,"load_error":"NameError","cases":[]}
    if "return List[int]" in code:
        return {"loaded":True,"load_error":None,
                "cases":[{"executed":False,"correct":False,"error":"NameError"} for _ in tests]}
    if "return int(x)" in code:
        return {"loaded":True,"load_error":None,
                "cases":[{"executed":True,"correct":True,"error":"none"} for _ in tests]}
    if "return 0" in code:
        return {"loaded":True,"load_error":None,
                "cases":[{"executed":True,"correct":i==0,"error":"none"} for i,_ in enumerate(tests)]}
    raise AssertionError("unexpected test completion")

def test_correct_and_wrong_are_distinguished(monkeypatch):
    monkeypatch.setattr(reward,"docker_probe",fake_probe)
    right=reward.score("def solve(x): return int(x)",TESTS)
    wrong=reward.score("def solve(x): return 0",TESTS)
    assert right["reward"]==1.0
    assert wrong["reward"]==0.28
    assert wrong["executable_fraction"]==1.0
    assert wrong["semantic_fraction"]==0.1

def test_nameerror_is_not_rewarded_as_executability(monkeypatch):
    monkeypatch.setattr(reward,"docker_probe",fake_probe)
    x=reward.score("def solve(x): return List[int]",TESTS)
    assert x["reward"]==0.0
    assert x["executable_fraction"]==0
    assert x["dependency_fraction"]==0

def test_loadtime_undefined_annotation(monkeypatch):
    monkeypatch.setattr(reward,"docker_probe",fake_probe)
    x=reward.score("def solve(x: List[int]): return 1",TESTS)
    assert x["reward"]==0 and x["verdict"]=="NameError"

def test_invalid_source(monkeypatch):
    def must_not_run(*_args,**_kwargs):
        raise AssertionError("Invalid Python must never trigger Docker probe")
    monkeypatch.setattr(reward,"docker_probe",must_not_run)
    assert reward.score("def solve(:pass",TESTS)["reward"]==0
