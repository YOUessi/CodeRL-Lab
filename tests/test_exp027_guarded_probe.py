import json,subprocess
import pytest
from coderl_lab.train import exp027_guarded_probe as m

TESTS=[{'x':str(i),'expected':i} for i in range(10)]
CODE='def solve(x): return int(x)'


def result():return {'loaded':True,'load_error':None,'cases':[{'executed':True,'correct':True,'error':'none'} for _ in TESTS]}


def fake(monkeypatch,behavior,cleanup_fail=False):
    calls=[]
    def run(cmd,**kw):
        calls.append(cmd)
        if cmd[1]=='rm':return subprocess.CompletedProcess(cmd,1 if cleanup_fail else 0,'','bad cleanup' if cleanup_fail else '')
        assert m.IMAGE_ID in cmd and '--name' in cmd and '--read-only' in cmd
        assert cmd[cmd.index('--network')+1]=='none'
        if behavior=='timeout':raise subprocess.TimeoutExpired(cmd,kw['timeout'])
        if behavior=='nonzero':return subprocess.CompletedProcess(cmd,137,'','')
        if behavior=='bad':return subprocess.CompletedProcess(cmd,0,'invalid','')
        return subprocess.CompletedProcess(cmd,0,'__EXP024__'+json.dumps(result()),'')
    monkeypatch.setattr(m.subprocess,'run',run)
    return calls


def test_valid_execution_and_cleanup(monkeypatch):
    calls=fake(monkeypatch,'ok')
    assert m.docker_probe(CODE,TESTS)==result()
    assert calls[-1][1:3]==['rm','-f']

@pytest.mark.parametrize('mode',['timeout','nonzero','bad'])
def test_indeterminate_does_not_become_zero_reward(monkeypatch,mode):
    calls=fake(monkeypatch,mode)
    with pytest.raises(m.SandboxIndeterminate):m.docker_probe(CODE,TESTS)
    assert calls[-1][1:3]==['rm','-f']
    assert calls[0][calls[0].index('--name')+1]==calls[-1][-1]


def test_cleanup_failure_blocks_success(monkeypatch):
    fake(monkeypatch,'ok',True)
    with pytest.raises(m.SandboxIndeterminate,match='cleanup'):m.docker_probe(CODE,TESTS)


def test_malformed_case_count_rejected():
    r=result();r['cases'].pop()
    with pytest.raises(m.SandboxIndeterminate):m.parse_verdict('__EXP024__'+json.dumps(r),TESTS)


def test_duplicate_verdict_rejected():
    s='__EXP024__'+json.dumps(result())
    with pytest.raises(m.SandboxIndeterminate):m.parse_verdict(s+'\n'+s,TESTS)


def test_wall_guard_validation():
    with pytest.raises(ValueError):m.docker_probe(CODE,TESTS,wall_seconds=0)
