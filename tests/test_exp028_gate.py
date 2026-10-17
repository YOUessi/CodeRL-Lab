import importlib.util
import pytest


def module():
    assert importlib.util.find_spec('coderl_lab.analysis.exp028_protocol'), 'EXP028 protocol implementation missing'
    from coderl_lab.analysis import exp028_protocol
    return exp028_protocol


def records(mixed=8):
    return [{'task_id':f'task/{i}','sample_id':j,'reward':float(j>0) if i<mixed else 1.,
             'syntax_valid':True,'clipped':False,'is_new':i<24,'semantic_fraction':float(j>0)}
            for i in range(48) for j in range(4)]


def test_gate_requires_signal_in_both_arms():
    m=module();r=m.gate({'narrow':records(8),'broad':records(8)})
    assert r['passed'] and r['arms']['broad']['new_mixed_groups']==8
    r=m.gate({'narrow':records(7),'broad':records(8)})
    assert not r['passed']


def test_broad_new_groups_need_signal():
    a=records(0)
    for r in a:
        if int(r['task_id'].split('/')[-1])>=24:r['reward']=float(r['sample_id']>0)
    assert not module().gate({'narrow':records(),'broad':a})['passed']


def test_duplicate_does_not_satisfy_gate():
    a=records();a[-1]=a[0]
    with pytest.raises(ValueError):module().gate({'narrow':a,'broad':records()})


def test_clipping_and_syntax_stop_training():
    a=records()
    for r in a[:40]:r['clipped']=True
    assert not module().gate({'narrow':a,'broad':records()})['passed']
    a=records()
    for r in a[:20]:r['syntax_valid']=False
    assert not module().gate({'narrow':a,'broad':records()})['passed']


def test_nan_reward_rejected():
    a=records();a[0]['reward']=float('nan')
    with pytest.raises(ValueError):module().gate({'narrow':a,'broad':records()})
