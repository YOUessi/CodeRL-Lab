"""CPU contracts; real Docker integration is separate on Tang."""
import importlib
import importlib.util
import json
from pathlib import Path
import pytest

MOD='coderl_lab.train.exp027_matched_reward'

def module():
    assert importlib.util.find_spec(MOD) is not None, 'matched reward implementation missing'
    return importlib.import_module(MOD)

def probe(error='none', correct=False):
    return {'loaded':True,'load_error':None,'cases':[{'executed':error=='none','correct':correct,'error':error} for _ in range(10)]}

TESTS=[{'x':str(i),'expected':i} for i in range(10)]

@pytest.mark.parametrize('error',['NameError','UnboundLocalError','ImportError','ModuleNotFoundError'])
def test_exception_family_is_only_intervention(error):
    m=module();a=m.rewards(probe(error))
    assert a['corrected_reward']==0
    assert a['legacy_reward']==(0 if error=='NameError' else .1)
    assert a['executable_fraction']==a['semantic_fraction']==0

def test_wrong_answer_not_full_score():
    m=module();a=m.rewards(probe())
    assert a['legacy_reward']==a['corrected_reward']==.2
    b=m.rewards(probe(correct=True))
    assert b['legacy_reward']==b['corrected_reward']==1

def test_load_error_zero_both():
    a=module().rewards({'loaded':False,'load_error':'NameError','cases':[]})
    assert a['legacy_reward']==a['corrected_reward']==0

def test_bad_verdict_rejected():
    p=probe();p['cases'][0]['correct']=True;p['cases'][0]['executed']=False
    with pytest.raises(ValueError): module().rewards(p)

def test_same_candidate_shares_real_observation(tmp_path,monkeypatch):
    m=module();calls=[]
    def observe(code,tests,wall_seconds):
        calls.append((code,tests,wall_seconds));return probe()
    monkeypatch.setattr(m.guarded,'docker_probe',observe)
    a=m.observe(tmp_path,'def solve(x): return 0',TESTS)
    b=m.observe(tmp_path,'def solve(x): return 0',TESTS)
    assert len(calls)==1 and calls[0][2]==60
    assert a['cache_hit'] is False and b['cache_hit'] is True
    assert a['key']==b['key'] and a['probe']==b['probe']

def test_different_tests_not_same_cache(tmp_path,monkeypatch):
    m=module();calls=[]
    monkeypatch.setattr(m.guarded,'docker_probe',lambda *a,**k:(calls.append(1) or probe()))
    a=m.observe(tmp_path,'def solve(x): return 0',TESTS)
    t=[dict(x) for x in TESTS];t[0]['expected']=42
    b=m.observe(tmp_path,'def solve(x): return 0',t)
    assert len(calls)==2 and a['key']!=b['key']

def test_timeout_never_becomes_zero_or_cache(tmp_path,monkeypatch):
    m=module()
    def fail(*a,**k):raise m.guarded.SandboxIndeterminate('wall guard')
    monkeypatch.setattr(m.guarded,'docker_probe',fail)
    with pytest.raises(m.guarded.SandboxIndeterminate):m.observe(tmp_path,'def solve(x): return 0',TESTS)
    assert not list(tmp_path.glob('*.json'))

def test_cache_tampering_fails(tmp_path,monkeypatch):
    m=module();monkeypatch.setattr(m.guarded,'docker_probe',lambda *a,**k:probe())
    a=m.observe(tmp_path,'def solve(x): return 0',TESTS)
    p=tmp_path/(a['key']+'.json');d=json.loads(p.read_text());d['probe']['cases'][0]['correct']=True;p.write_text(json.dumps(d))
    with pytest.raises(ValueError):m.observe(tmp_path,'def solve(x): return 0',TESTS)

def test_metadata_and_partial_group_fail_closed(tmp_path,monkeypatch):
    m=module();counter=[0]
    def varying(*a,**k):
        counter[0]+=1
        if counter[0]==2:raise m.guarded.SandboxIndeterminate('no score')
        return probe()
    monkeypatch.setattr(m.guarded,'docker_probe',varying)
    log=tmp_path/'run'/'reward_attempts.jsonl';f=m.reward_factory(log,'legacy',tmp_path/'cache')
    with pytest.raises(m.guarded.SandboxIndeterminate):
        f([[]]*4,['def solve(x): return '+str(i) for i in range(4)],task_id=['q']*4,test_cases=[TESTS]*4)
    assert not log.exists() or not log.read_text().strip()

def test_invalid_python_no_container(tmp_path,monkeypatch):
    m=module()
    def fail(*a,**k):raise AssertionError('invalid source executed')
    monkeypatch.setattr(m.guarded,'docker_probe',fail)
    f=m.reward_factory(tmp_path/'r'/'reward_attempts.jsonl','corrected',tmp_path/'cache')
    assert f([[]]*4,['def solve(: pass']*4,task_id=['q']*4,test_cases=[TESTS]*4)==[0]*4
