"""EXP-024 strict reward unit tests, candidates only run in Docker."""
from coderl_lab.train.exp024_reward import score

TESTS=[{"x":str(i),"expected":i} for i in range(10)]

def test_correct_and_wrong_are_distinguished():
    right=score("def solve(x): return int(x)",TESTS)
    wrong=score("def solve(x): return 0",TESTS)
    assert right["reward"]==1.0
    assert wrong["reward"]==0.28
    assert wrong["executable_fraction"]==1.0
    assert wrong["semantic_fraction"]==0.1

def test_nameerror_is_not_rewarded_as_executability():
    x=score("def solve(x): return List[int]",TESTS)
    assert x["reward"]==0.0
    assert x["executable_fraction"]==0
    assert x["dependency_fraction"]==0

def test_loadtime_undefined_annotation():
    x=score("def solve(x: List[int]): return 1",TESTS)
    assert x["reward"]==0 and x["load_error"]=="NameError"

def test_invalid_source():
    assert score("def solve(:pass",TESTS)["reward"]==0
