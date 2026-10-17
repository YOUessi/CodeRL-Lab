"""CPU reward contracts: test actual scoring, replace only Docker transport.

Real pinned-container checks run explicitly on Tang. Hosted-runner wall-clock
startup must not turn this unit test into an intermittent integration test.
"""
import pytest
from coderl_lab.train.exp023_reward_arms import ablation_score
from coderl_lab.train import train014b_reward_sandbox as sandbox

CASES=[{'x':str(i),'expected':i} for i in range(10)]


def test_format_independent_of_semantics(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('format reward must not execute Docker')
    monkeypatch.setattr(sandbox,'docker_test',forbidden)
    assert ablation_score('def solve(x): return int(x)',CASES,'format')['reward']==1
    assert ablation_score('def solve(x): return 0',CASES,'format')['reward']==1
    assert ablation_score('def solve(: pass',CASES,'format')['reward']==0


def test_reward_arms_are_distinct(monkeypatch):
    calls=[]
    def fixed_probe(code,cases):
        assert code=='def solve(x): return 0'
        assert cases==CASES
        calls.append(code)
        return {'correct':False,'passed':1,'error':'wrong_answer'}
    monkeypatch.setattr(sandbox,'docker_test',fixed_probe)
    assert ablation_score('def solve(x): return 0',CASES,'full')['reward']==.19
    assert ablation_score('def solve(x): return 0',CASES,'execution')['reward']==1
    assert ablation_score('def solve(x): return 0',CASES,'format')['reward']==1
    assert len(calls)==2


def test_missing_entry_has_no_reward(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('missing entry must not execute Docker')
    monkeypatch.setattr(sandbox,'docker_test',forbidden)
    for arm in ('format','execution','full'):
        assert ablation_score('def f(x): return x',CASES,arm)['reward']==0


def test_infrastructure_error_propagates(monkeypatch):
    def broken(*args,**kwargs):raise RuntimeError('checker unavailable')
    monkeypatch.setattr(sandbox,'docker_test',broken)
    for arm in ('full','execution'):
        with pytest.raises(RuntimeError,match='checker unavailable'):
            ablation_score('def solve(x): return 0',CASES,arm)
