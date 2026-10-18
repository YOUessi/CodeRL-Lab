"""CPU-only contracts; Docker subprocesses are fixtures, real integration runs on Tang."""
import importlib.util
import json
from types import SimpleNamespace
import pytest


def module():
    name='coderl_lab.analysis.exp029_frozen_eval'
    assert importlib.util.find_spec(name) is not None, 'missing EXP029 implementation'
    return __import__(name,fromlist=['*'])


def oracle():
    return {'task_id':'q','entry_point':'f','test':'assert f(0)==0','canonical_code':'def f(x): return x'}


def fixture_run(m,monkeypatch,rc=0,oom=False,stdout=None,timeout=False):
    calls=[]
    state={'Running':False,'ExitCode':rc,'OOMKilled':oom,'Dead':False,'Error':''}
    def run(cmd,**kw):
        calls.append(cmd)
        if cmd[1]=='run':
            if timeout:raise m.subprocess.TimeoutExpired(cmd,60)
            text='__EXP019B__'+json.dumps({'passed':True,'reason':'none'}) if stdout is None else stdout
            return SimpleNamespace(returncode=rc,stdout=text,stderr='')
        if cmd[1]=='inspect':return SimpleNamespace(returncode=0,stdout=json.dumps(state),stderr='')
        if cmd[1:3]==['rm','-f']:return SimpleNamespace(returncode=0,stdout='',stderr='')
        raise AssertionError(cmd)
    monkeypatch.setattr(m.subprocess,'run',run)
    monkeypatch.setattr(m,'ensure_host_capacity',lambda:None)
    return calls


def test_protocol_is_uniform_not_per_model():
    m=module();p=m.protocol()
    assert p['memory_bytes']==4*1024**3 and p['swap_bytes']==0
    assert p['cpu_seconds_per_process']==15 and p['host_wall_guard_seconds']==60
    assert p['image_id']==m.ev.IMAGE_ID and p['driver_sha256']==m.ev.protocol()['driver_sha256']
    assert p['different_from_historical_1GiB'] is True


def test_host_capacity_fail_closed(tmp_path):
    m=module();p=tmp_path/'meminfo'
    p.write_text('MemAvailable: 100 kB\n')
    with pytest.raises(m.SandboxIndeterminate):m.ensure_host_capacity(p)
    p.write_text('MemAvailable: 16000000 kB\n');m.ensure_host_capacity(p)


def test_real_command_constraints_and_cache(tmp_path,monkeypatch):
    m=module();calls=fixture_run(m,monkeypatch)
    a=m.score('def f(x): return x',oracle(),tmp_path)
    b=m.score('def f(x): return x',oracle(),tmp_path)
    cmd=calls[0]
    assert cmd[cmd.index('--memory')+1]=='4g'
    assert cmd[cmd.index('--memory-swap')+1]=='4g'
    assert cmd[cmd.index('--network')+1]=='none'
    assert '--read-only' in cmd and '--cap-drop' in cmd
    assert a['passed'] and not a['cache_hit'] and b['cache_hit']
    assert sum(x[1]=='run' for x in calls)==1
    assert calls[-1][1:3]==['rm','-f']
    evidence=json.loads(next(tmp_path.glob('*.complete.json')).read_text())
    assert evidence['cleanup_confirmed'] is True and evidence['container_state']['Running'] is False


def test_different_test_is_not_cache_hit(tmp_path,monkeypatch):
    m=module();calls=fixture_run(m,monkeypatch)
    m.score('def f(x): return x',oracle(),tmp_path)
    o=oracle();o['test']='assert f(1)==1'
    assert not m.score('def f(x): return x',o,tmp_path)['cache_hit']
    assert sum(x[1]=='run' for x in calls)==2


def test_cache_tampering_rejected(tmp_path,monkeypatch):
    m=module();fixture_run(m,monkeypatch)
    m.score('def f(x): return x',oracle(),tmp_path)
    p=next(tmp_path.glob('*.complete.json'));x=json.loads(p.read_text());x['passed']=False;p.write_text(json.dumps(x))
    with pytest.raises(ValueError):m.score('def f(x): return x',oracle(),tmp_path)


def test_wall_timeout_has_no_score_and_cleans(tmp_path,monkeypatch):
    m=module();calls=fixture_run(m,monkeypatch,timeout=True)
    with pytest.raises(m.SandboxIndeterminate):m.score('def f(x): return x',oracle(),tmp_path)
    assert not list(tmp_path.glob('*.complete.json'))
    assert list(tmp_path.glob('*.failed.json')) and list(tmp_path.glob('*.prepared.json'))
    assert calls[-1][1:3]==['rm','-f']
    n=len(calls)
    with pytest.raises(RuntimeError):m.score('def f(x): return x',oracle(),tmp_path)
    assert len(calls)==n


@pytest.mark.parametrize('rc,oom,out,reason',[
    (137,True,'','memory_budget_exceeded'),
    (124,False,'__EXP027_CPU_BUDGET__\n','cpu_budget_exceeded')])
def test_verified_resource_failure(tmp_path,monkeypatch,rc,oom,out,reason):
    m=module();fixture_run(m,monkeypatch,rc,oom,out)
    x=m.score('def f(x): return x',oracle(),tmp_path)
    assert not x['passed'] and x['resource_limited'] and x['reason']==reason


def test_unconfirmed_137_not_zero_score(tmp_path,monkeypatch):
    m=module();fixture_run(m,monkeypatch,137,False,'')
    with pytest.raises(m.SandboxIndeterminate):m.score('def f(x): return x',oracle(),tmp_path)
    assert not list(tmp_path.glob('*.complete.json'))


def test_bad_python_never_executes(tmp_path,monkeypatch):
    m=module();calls=fixture_run(m,monkeypatch)
    r=m.score('def f(',oracle(),tmp_path)
    assert r['reason']=='invalid_syntax_or_entry' and not calls and not r['container_executed']


def test_reference_gate_needs_all_96():
    m=module();ids={f'q{i}' for i in range(96)}
    rr=[{'task_id':q,'passed':True,'reason':'none'} for q in sorted(ids)]
    assert m.control_gate(rr,ids)['model_scoring_permitted']
    with pytest.raises(ValueError):m.control_gate(rr[:-1],ids)
    rr[0].update(passed=False,reason='memory_budget_exceeded')
    x=m.control_gate(rr,ids)
    assert x['total']==96 and x['failed']==1 and not x['model_scoring_permitted']


def test_casewise_pairing_not_candidate_independence():
    m=module();a={'q':[True,False,False,False]};b={'q':[True,True,False,False]}
    x=m.ev.compare(a,b,reps=100)
    assert x['candidate_delta']==.25 and x['resample_unit']=='task' and x['repairs']==1
