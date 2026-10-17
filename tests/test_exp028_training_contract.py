import importlib.util
import pytest


def module():
    assert importlib.util.find_spec('coderl_lab.train.exp028_7b_grpo'), 'EXP028 training wrapper missing'
    from coderl_lab.train import exp028_7b_grpo
    return exp028_7b_grpo


def config():
    return {'experiment':{'seed':43,'coverage_arm':'narrow','reward_arm':'strict'},
            'grpo':{'num_generations':4,'per_device_train_batch_size':4,'gradient_accumulation_steps':1,
                    'max_completion_length':256,'learning_rate':1e-5,'temperature':.8,'top_p':.95,'beta':0.,'loss_type':'grpo'},
            'runs':{'formal_steps':32},'reward':{'wall_guard_seconds':60.}}


def test_fixed_training_contract():
    assert module().validate_controls(config(),'narrow') is None


@pytest.mark.parametrize('field,value',[('num_generations',8),('learning_rate',2e-5),('max_completion_length',512),('beta',.1)])
def test_budget_change_rejected(field,value):
    c=config();c['grpo'][field]=value
    with pytest.raises(ValueError):module().validate_controls(c,'narrow')


def test_coverage_seed_mismatch():
    with pytest.raises(ValueError):module().validate_controls(config(),'broad')
    c=config();c['experiment']['seed']=99
    with pytest.raises(ValueError):module().validate_controls(c,'narrow')
