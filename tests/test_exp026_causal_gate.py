import pytest
from coderl_lab.analysis.exp026_causal_gate import compare_trace


def row(i,reward=.5,verdict='completed',e=1.,d=1.,s=.375,code=None,legacy=None):
    r={'task_id':'t','code_sha256':code or str(i),'reward':reward,'verdict':verdict,
       'executable_fraction':e,'dependency_fraction':d,'semantic_fraction':s}
    if legacy is not None:r.update(legacy_reward_same_probe=legacy,legacy_dependency_fraction=d)
    return r


def test_identical_trace_is_not_confounded():
    a=[row(i) for i in range(4)];b=[row(i,legacy=.5) for i in range(4)]
    assert compare_trace(a,b)['status']=='no_observed_same_code_execution_drift'


def test_same_code_timeout_blocks_clean_attribution():
    a=[row(i) for i in range(4)];b=[row(i,legacy=.5) for i in range(4)]
    b[1]=row(1,reward=0.,verdict='timeout',e=0.,d=0.,s=0.,legacy=0.)
    r=compare_trace(a,b)
    assert r['status']=='runtime_confounded'
    assert r['same_code_probe_drift'][0]['attempt_id']==1
    assert r['classification_effect_count']==0


def test_classification_change_is_separate_from_probe_change():
    a=[row(i) for i in range(4)];b=[row(i,legacy=.5) for i in range(4)]
    b[2]['dependency_fraction']=0.;b[2]['reward']=.4
    r=compare_trace(a,b)
    assert r['classification_effect_count']==1
    assert r['status']=='no_observed_same_code_execution_drift'


@pytest.mark.parametrize('mismatch',['length','task','duplicate'])
def test_malformed_trace_rejected(mismatch):
    a=[row(i) for i in range(4)];b=[row(i,legacy=.5) for i in range(4)]
    if mismatch=='length':b.pop()
    elif mismatch=='task':b[2]['task_id']='other'
    else:
        for r in b:r['attempt_id']=0
    with pytest.raises(ValueError):compare_trace(a,b)
