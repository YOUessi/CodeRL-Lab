"""No Docker in CPU unit tests; probe integration is validated on Tang."""
import importlib,importlib.util,json,subprocess
import pytest

def module():
    n='coderl_lab.analysis.exp027_matched_eval'
    assert importlib.util.find_spec(n) is not None,'matched evaluator missing'
    return importlib.import_module(n)

def test_budget_exit_is_distinct_from_host_failure():
    m=module()
    assert m.decode_exit(124,'__EXP027_CPU_BUDGET__\n')['reason']=='cpu_budget_exceeded'
    with pytest.raises(m.SandboxIndeterminate):m.decode_exit(124,'')
    with pytest.raises(m.SandboxIndeterminate):m.decode_exit(137,'')

def test_regular_verdict_and_malformed_result():
    m=module()
    assert m.decode_exit(0,'__EXP019B__'+json.dumps({'passed':True,'reason':'none'}))['passed'] is True
    with pytest.raises(m.SandboxIndeterminate):m.decode_exit(0,'broken')
    with pytest.raises(m.SandboxIndeterminate):m.decode_exit(0,'__EXP019B__'+json.dumps({'passed':1,'reason':'none'}))

def test_inference_keys_checked():
    m=module();s={'configuration':dict(m.GENERATION),'public_sha256':m.PUBLIC_SHA,'model_revision':m.REVISION,
        'model_inference_adapter_dtype':{'sft':['torch.float32'],'grpo':['torch.float32']}}
    m.validate_recipe(s)
    s['configuration']['top_p']=.5
    with pytest.raises(ValueError):m.validate_recipe(s)

def test_paired_orientation_and_no_false_independent_candidates():
    m=module();a={'q':[False,True,False,False],'r':[False]*4};b={'q':[True,True,False,False],'r':[False]*4}
    x=m.compare(a,b,reps=100)
    assert x['candidate_delta']==.125 and x['repairs']==1 and x['regressions']==0
    assert x['resample_unit']=='task' and x['task_count']==2

def test_resource_sample_kept_in_full_denominator():
    m=module();rows=[{'task_id':'q','sample_id':i,'passed':i==0,'reason':'none' if i==0 else 'cpu_budget_exceeded'} for i in range(4)]
    x=m.metrics(rows)
    assert x['correct']==1 and x['candidate_total']==4 and x['reasons']['cpu_budget_exceeded']==3
