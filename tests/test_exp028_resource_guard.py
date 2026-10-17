import importlib.util,json,subprocess
from types import SimpleNamespace
import pytest


def module():
    assert importlib.util.find_spec('coderl_lab.train.exp028_resource_guard'), 'EXP028 OOM classifier missing'
    from coderl_lab.train import exp028_resource_guard
    return exp028_resource_guard


def test_confirmed_oom_only():
    m=module()
    r=m.resource_exit(137,{'OOMKilled':True,'ExitCode':137,'Running':False})
    assert not r['loaded'] and r['load_error']=='memory_budget_exceeded'
    for state in ({'OOMKilled':False,'ExitCode':137,'Running':False},{'OOMKilled':True,'ExitCode':137,'Running':True},{}):
        with pytest.raises(m.SandboxIndeterminate):m.resource_exit(137,state)
    with pytest.raises(m.SandboxIndeterminate):m.resource_exit(1,{'OOMKilled':True,'ExitCode':137,'Running':False})


def test_container_oom_requires_inspect_and_cleanup(monkeypatch):
    m=module();calls=[]
    def fake(args,**kwargs):
        calls.append(args)
        if args[1]=='run':return SimpleNamespace(returncode=137,stdout='',stderr='')
        if args[1]=='inspect':return SimpleNamespace(returncode=0,stdout=json.dumps({'OOMKilled':True,'ExitCode':137,'Running':False}),stderr='')
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(m.subprocess,'run',fake)
    r=m.docker_probe('def solve(x): return 0',[{'x':str(i),'expected':0} for i in range(10)])
    assert r['load_error']=='memory_budget_exceeded'
    assert [c[1] for c in calls]==['run','inspect','rm']


def test_wall_timeout_aborts_and_cleans(monkeypatch):
    m=module();calls=[]
    def fake(args,**kwargs):
        calls.append(args)
        if args[1]=='run':raise subprocess.TimeoutExpired(args,60)
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(m.subprocess,'run',fake)
    with pytest.raises(m.SandboxIndeterminate):m.docker_probe('def solve(x): return 0',[{'x':str(i),'expected':0} for i in range(10)])
    assert calls[-1][1]=='rm'


def test_unknown_137_never_becomes_reward(monkeypatch):
    m=module()
    def fake(args,**kwargs):
        if args[1]=='run':return SimpleNamespace(returncode=137,stdout='',stderr='')
        if args[1]=='inspect':return SimpleNamespace(returncode=0,stdout=json.dumps({'OOMKilled':False,'ExitCode':137,'Running':False}),stderr='')
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(m.subprocess,'run',fake)
    with pytest.raises(m.SandboxIndeterminate):m.docker_probe('def solve(x): return 0',[{'x':str(i),'expected':0} for i in range(10)])
