import math
import pytest
from coderl_lab.analysis.exp026_core import features, advantage, groups, ablation


def make(error='none', executed=True, correct=False):
    return {'loaded':True,'load_error':None,'cases':[{'executed':executed,'correct':correct,'error':error} for _ in range(4)]}

@pytest.mark.parametrize('error',['NameError','UnboundLocalError','ImportError','ModuleNotFoundError'])
def test_dependency_errors_not_awarded(error):
    f=features(make(error,False))
    assert f['dependency_fraction']==0
    assert f['reward']==0


def test_success_and_wrong_are_distinguished():
    assert features(make(correct=True))['reward']==1
    assert features(make())['reward']==.2


def test_unrelated_runtime_error_not_claimed_dependency_failure():
    f=features(make('ZeroDivisionError',False))
    assert f['executable_fraction']==0
    assert f['dependency_fraction']==1
    assert f['reward']==.1


def test_explicit_dependency_subclass_flag():
    d=make('CustomMissingName',False)
    for r in d['cases']:r['dependency_error']=True
    assert features(d)['reward']==0

@pytest.mark.parametrize('bad',[
    {'loaded':True,'cases':[]},
    {'loaded':True,'cases':[{'executed':False,'correct':True,'error':'NameError'}]},
    {'loaded':'yes','cases':[]},
    {'loaded':True,'cases':[{'executed':1,'correct':False,'error':'none'}]},
])
def test_malformed_probe_fails_closed(bad):
    with pytest.raises(ValueError):features(bad)


def test_group_affine_invariance_without_epsilon():
    s=[.0,.2,.5,1.]
    assert advantage([.1+.9*x for x in s],eps=0)==pytest.approx(advantage([.2+.8*x for x in s],eps=0))
    assert advantage([1.,1.,1.,1.])==[0.,0.,0.,0.]


def test_invalid_group_membership_rejected():
    with pytest.raises(ValueError):list(groups([{'task_id':str(i),'reward':0.} for i in range(4)]))
    with pytest.raises(ValueError):list(groups([{'task_id':'x','reward':0.}]))


def test_ablation_constant_dependency_cancels():
    r=[{'task_id':'x','executable_fraction':1.,'dependency_fraction':1.,'semantic_fraction':s,'reward':round(.2+.8*s,6)} for s in [0.,.2,.5,1.]]
    out=ablation(r)
    assert out['varying_dependency_groups']==0
    assert out['remove_dependency']['sign_flips']==0
    assert out['remove_dependency']['max_advantage_delta']<1e-10


def test_ablation_does_not_rescale_remaining_terms():
    r=[{'task_id':'x','executable_fraction':1.,'dependency_fraction':d,'semantic_fraction':.5,'reward':round(.1+.1*d+.4,6)} for d in [0.,1.,1.,1.]]
    out=ablation(r)
    assert out['varying_dependency_groups']==1
    assert out['remove_dependency']['max_advantage_delta']>1
