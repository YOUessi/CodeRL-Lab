"""Reward arm contract tests: no GPU, no model download, Docker only for runtime proxy."""
from coderl_lab.train.exp023_reward_arms import ablation_score

CASES=[{"x":str(i),"expected":i} for i in range(10)]
def test_format_independent_of_semantics():
    assert ablation_score("def solve(x): return int(x)",CASES,"format")["reward"]==1
    assert ablation_score("def solve(x): return 0",CASES,"format")["reward"]==1
    assert ablation_score("def solve(: pass",CASES,"format")["reward"]==0

def test_reward_arms_are_distinct():
    assert ablation_score("def solve(x): return 0",CASES,"full")["reward"]==0.19
    assert ablation_score("def solve(x): return 0",CASES,"execution")["reward"]==1
    assert ablation_score("def solve(x): return 0",CASES,"format")["reward"]==1

def test_missing_entry_has_no_reward():
    for arm in ("format","execution","full"):
        assert ablation_score("def f(x): return x",CASES,arm)["reward"]==0
