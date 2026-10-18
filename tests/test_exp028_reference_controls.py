"""Reference audit contracts. CPU-only tests never run candidate code or Docker."""
import importlib.util
import json
from types import SimpleNamespace
import pytest


def mod():
    name='coderl_lab.analysis.exp028_reference_controls'
    assert importlib.util.find_spec(name) is not None, 'missing reference-control evidence implementation'
    return __import__(name,fromlist=['*'])


def state(rc=0,oom=False,running=False):
    return {'Running':running,'OOMKilled':oom,'ExitCode':rc,'Dead':False,'Error':''}


def test_confirmed_reference_oom_is_resource_failure():
    m=mod();r=m.classify_exit(137,'',state(137,True))
    assert r=={'passed':False,'reason':'memory_budget_exceeded','resource_limited':True}


@pytest.mark.parametrize('rc,s',[(137,state(137)),(137,state(137,True,True)),(137,state(0,True)),(1,state(1))])
def test_unknown_exit_is_not_a_score(rc,s):
    m=mod()
    with pytest.raises(m.SandboxIndeterminate):m.classify_exit(rc,'',s)


def test_cpu_marker_needs_matching_exit_and_state():
    m=mod();mark='__EXP027_CPU_BUDGET__\n'
    assert m.classify_exit(124,mark,state(124))['reason']=='cpu_budget_exceeded'
    with pytest.raises(m.SandboxIndeterminate):m.classify_exit(124,'',state(124))
    with pytest.raises(m.SandboxIndeterminate):m.classify_exit(124,mark,state(124,False,True))


def test_normal_reference_verdict():
    m=mod();out='__EXP019B__'+json.dumps({'passed':True,'reason':'none'})
    assert m.classify_exit(0,out,state())=={'passed':True,'reason':'none','resource_limited':False}
    with pytest.raises(m.SandboxIndeterminate):m.classify_exit(0,'',state())


def test_full_denominator_and_reference_stop():
    m=mod();rr=[{'task_id':'a','passed':True,'reason':'none','resource_limited':False},
               {'task_id':'b','passed':False,'reason':'memory_budget_exceeded','resource_limited':True}]
    s=m.control_summary(rr,{'a','b'})
    assert s['status']=='stopped_reference_controls' and s['passed']==1 and s['total']==2
    assert s['dropped_tasks']==0 and s['model_scoring_permitted'] is False
    assert s['failure_types']=={'memory_budget_exceeded':1}


def test_missing_or_duplicate_reference_is_not_complete():
    m=mod();r={'task_id':'a','passed':True,'reason':'none','resource_limited':False}
    with pytest.raises(ValueError):m.control_summary([r],{'a','b'})
    with pytest.raises(ValueError):m.control_summary([r,r],{'a','b'})


def test_all_controls_pass_allows_original_next_stage():
    m=mod();r={'task_id':'a','passed':True,'reason':'none','resource_limited':False}
    s=m.control_summary([r],{'a'})
    assert s['status']=='reference_controls_passed' and s['model_scoring_permitted'] is True


def test_wall_timeout_preserves_failure_and_cleans(tmp_path,monkeypatch):
    m=mod();calls=[]
    def fake(cmd,**kwargs):
        calls.append(cmd)
        if cmd[1]=='run':raise m.subprocess.TimeoutExpired(cmd,60)
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(m.subprocess,'run',fake)
    with pytest.raises(m.SandboxIndeterminate):
        m.execute_reference({'task_id':'a','canonical_code':'def f(): return 1','entry_point':'f','test':'assert f()==1'},tmp_path)
    assert calls[-1][1:3]==['rm','-f']
    assert list(tmp_path.glob('*.prepared.json')) and list(tmp_path.glob('*.failed.json'))
    assert not list(tmp_path.glob('*.complete.json'))
    cmd=calls[0];assert cmd[cmd.index('--memory')+1]=='1g' and '--network' in cmd


def test_engine_evidence_retained_before_cleanup(tmp_path,monkeypatch):
    m=mod();calls=[]
    def fake(cmd,**kwargs):
        calls.append(cmd)
        if cmd[1]=='inspect':return SimpleNamespace(returncode=0,stdout=json.dumps(state(137,True)),stderr='')
        return SimpleNamespace(returncode=137 if cmd[1]=='run' else 0,stdout='',stderr='')
    monkeypatch.setattr(m.subprocess,'run',fake)
    r=m.execute_reference({'task_id':'b','canonical_code':'def f(): return []','entry_point':'f','test':'f()'},tmp_path)
    assert r['reason']=='memory_budget_exceeded' and r['container_state']['OOMKilled'] is True
    assert r['cleanup_confirmed'] is True
    assert [c[1] for c in calls]==['run','inspect','rm']
    assert len(list(tmp_path.glob('*.complete.json')))==1
