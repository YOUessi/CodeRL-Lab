"""CPU-only checks; never download models, start Docker or allocate a GPU."""
import importlib.util
import pytest
from coderl_lab.train import exp026_reward as reward
from coderl_lab.train import exp026_7b_grpo as wrapper
from coderl_lab.train.gpu_preflight import classify_gpu_compute_processes


def test_clean_checkout_contains_entire_chain():
    for name in ('coderl_lab.train.gpu_preflight','coderl_lab.train.exp026_7b_grpo',
                 'coderl_lab.analysis.exp026_runner','coderl_lab.train.exp024_7b_grpo'):
        assert importlib.util.find_spec(name) is not None


def test_unknown_gpu_compute_process_blocks():
    d=classify_gpu_compute_processes('1234, /opt/python, 512\n')
    assert not d['safe_for_single_model_job']
    assert d['blocking_processes'][0]['pid']=='1234'


def test_display_daemon_does_not_block():
    assert classify_gpu_compute_processes('12, /usr/bin/Xorg, 20\n')['safe_for_single_model_job']
    with pytest.raises(ValueError):classify_gpu_compute_processes('bad, data\n')


def test_wrapper_restores_training_globals_after_failure(monkeypatch):
    b=wrapper.baseline;old_reward=b.reward_factory;old_preflight=b.preflight
    def stopped():
        assert b.reward_factory is wrapper.corrected.reward_factory
        raise RuntimeError('simulated preflight failure')
    monkeypatch.setattr(b,'main',stopped)
    with pytest.raises(RuntimeError,match='simulated'):
        wrapper.main()
    assert b.reward_factory is old_reward
    assert b.preflight is old_preflight


def test_same_probe_counterfactual_exposes_previously_missed_dependency(monkeypatch):
    def fake(code,tests):
        return {'loaded':True,'load_error':None,'cases':[
            {'executed':False,'correct':False,'error':'ModuleNotFoundError'} for _ in tests]}
    monkeypatch.setattr(reward,'docker_probe',fake)
    r=reward.score('def solve(x): return 0',[{'x':str(i),'expected':i} for i in range(10)])
    assert r['reward']==0
    assert r['legacy_reward_same_probe']==.1
    assert r['legacy_dependency_fraction']==1
    assert r['dependency_fraction']==0
