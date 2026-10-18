"""CPU-only contracts. No real Docker or GPU required in CI."""
import importlib.util
import json
from types import SimpleNamespace
import pytest


def module(name):
    full='coderl_lab.train.'+name
    assert importlib.util.find_spec(full) is not None, 'missing implementation '+full
    return __import__(full,fromlist=['*'])


def completed():
    return {'loaded':True,'load_error':None,'cases':[{'executed':True,'correct':True,'error':'none'}]*10}


def test_confirmed_cpu_budget_is_resource_failure():
    m=module('exp028_cpu_guard')
    x=m.decode_result(124,m.CPU_MARKER+'\n',{'Running':False,'ExitCode':124,'OOMKilled':False},[{}]*10)
    assert x['load_error']=='cpu_budget_exceeded' and x['resource_limited'] is True


@pytest.mark.parametrize('rc,out,state',[(124,'',{'Running':False,'ExitCode':124}),(124,'__EXP028_CPU_BUDGET__\n',{'Running':True,'ExitCode':0}),(137,'',{'Running':False,'ExitCode':137,'OOMKilled':False}),(1,'',{})])
def test_unknown_exits_not_rewards(rc,out,state):
    m=module('exp028_cpu_guard')
    with pytest.raises(m.SandboxIndeterminate):m.decode_result(rc,out,state,[{}]*10)


def test_confirmed_oom_preserved():
    m=module('exp028_cpu_guard')
    r=m.decode_result(137,'',{'Running':False,'ExitCode':137,'OOMKilled':True},[{}]*10)
    assert r['load_error']=='memory_budget_exceeded'


def test_normal_verdict_and_missing_verdict():
    m=module('exp028_cpu_guard')
    assert m.decode_result(0,'__EXP024__'+json.dumps(completed()),{},[{}]*10)['loaded']
    with pytest.raises(m.SandboxIndeterminate):m.decode_result(0,'',{},[{}]*10)


def test_wall_timeout_cleans_without_reward(monkeypatch):
    m=module('exp028_cpu_guard');calls=[]
    def fake(cmd,**kw):
        calls.append(cmd)
        if cmd[1]=='run':raise m.subprocess.TimeoutExpired(cmd,60)
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(m.subprocess,'run',fake)
    with pytest.raises(m.SandboxIndeterminate):m.docker_probe('def solve(x): return x',[{'x':i,'expected':i} for i in range(10)])
    assert calls[-1][1:3]==['rm','-f']
    assert '--memory' in calls[0] and calls[0][calls[0].index('--memory')+1]=='256m'


def test_pending_group_survives_failure_and_locates_code(tmp_path,monkeypatch):
    m=module('exp028_durable_reward')
    def bad_observe(*args,**kwargs):raise RuntimeError('test_probe_failed')
    monkeypatch.setattr(m.matched,'observe',bad_observe)
    fn=m.reward_factory(tmp_path/'reward_attempts.jsonl','corrected',tmp_path/'cache')
    codes=['def solve(x): return int(x)']*4;tests=[[{'x':i,'expected':i} for i in range(10)]]*4
    with pytest.raises(RuntimeError,match='test_probe_failed'):fn(['p']*4,codes,task_id=['t']*4,test_cases=tests)
    saved=json.loads((tmp_path/'pending_groups/group_0000.json').read_text())
    assert len(saved['completions'])==4 and saved['status']=='prepared_before_execution'
    events=[json.loads(x) for x in (tmp_path/'probe_events.jsonl').read_text().splitlines()]
    assert events[0]['status']=='started' and events[-1]['status']=='failed'
    assert events[-1]['code_sha256']==events[0]['code_sha256']
    assert not (tmp_path/'reward_attempts.jsonl').exists()
    assert m.matched.observe is bad_observe


def test_complete_group_retains_original_reward(tmp_path,monkeypatch):
    m=module('exp028_durable_reward')
    monkeypatch.setattr(m.matched,'observe',lambda *args,**kw: {'probe':completed(),'key':'key','probe_sha256':'sha','cache_hit':False})
    fn=m.reward_factory(tmp_path/'reward_attempts.jsonl','corrected',tmp_path/'cache')
    vals=fn(['p']*4,['def solve(x): return int(x)']*4,task_id=['t']*4,test_cases=[[{'x':i,'expected':i} for i in range(10)]]*4)
    assert vals==[1.0]*4
    assert len((tmp_path/'reward_attempts.jsonl').read_text().splitlines())==4


def test_incomplete_metadata_has_no_prepared_group(tmp_path):
    m=module('exp028_durable_reward');f=m.reward_factory(tmp_path/'reward_attempts.jsonl','corrected',tmp_path/'cache')
    with pytest.raises(ValueError):f(['p'],['pass'],task_id=['t'],test_cases=[[]])
    assert not (tmp_path/'pending_groups').exists()
