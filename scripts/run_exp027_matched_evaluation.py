"""Finish EXP027 without further manual phase handoffs.

Exact-weight/exact-recipe generation reuse is explicit. All verdicts are newly
computed under the one current protocol (deduplicated by exact code+test key).
No private benchmark tests are loaded before the complete generation freeze.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from coderl_lab.analysis import exp027_matched_eval as ev
from coderl_lab.analysis.exp021_seed42_generate import sample_seed
from coderl_lab.train.exp027_matched_reward import file_sha,canonical,protocol
from run_exp027_matched_training import verify_run

OLD_SHA='47e0d8d6687efe1965c2ba7ff88f064b56fe3d10907c742824b116abe1ffc8b4'

def once(p,value):
    text=canonical(value)+'\n'
    if p.exists():
        if p.read_text()!=text:raise ValueError('immutable artifact changed: '+str(p))
    else:
        p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x') as f:f.write(text)

def normal_records(rr,ids):
    found={}
    for r in rr:
        key=(r['task_id'],r['sample_id'])
        if key in found or key[0] not in ids or key[1] not in range(4):raise ValueError('candidate keys changed')
        if r['seed']!=sample_seed(*key) or ev.objsha(r['raw_code'])==None:raise ValueError('sampling identity mismatch')
        import hashlib
        if r['code_sha256']!=hashlib.sha256(r['raw_code'].encode()).hexdigest():raise ValueError('candidate hash mismatch')
        found[key]={k:r[k] for k in ('task_id','sample_id','seed','raw_code','code_sha256','generated_tokens')}
    expected={(q,i) for q in ids for i in range(4)}
    if set(found)!=expected:raise ValueError('not 384 complete outputs')
    return [found[k] for k in sorted(found)]

def from_bundle(path,arm,weight,ids):
    sp=path/'generation_summary.json'
    if not sp.exists():return None
    s=json.loads(sp.read_text())
    if s.get('model_adapters_sha256',{}).get(arm)!=weight:return None
    ev.validate_recipe(s)
    if s['status']!='all_samples_generated_frozen' or s['checkpoint_sha256']!=ev.sha(path/'checkpoint.jsonl'):
        raise ValueError('source generation not frozen')
    data=normal_records([x for x in ev.rows(path/'checkpoint.jsonl') if x['arm']==arm],ids)
    return data,{'kind':'exact_weight_recipe_reuse','source':str(path),'checkpoint_sha256':s['checkpoint_sha256'],'arm':arm}

def write_generation(path,data):
    text=''.join(canonical(x)+'\n' for x in data)
    if path.exists():
        if path.read_text()!=text:raise ValueError('generation drift')
    else:
        with path.open('x') as f:f.write(text)

def generate_new(work,data_root,model,weight,out,sft_rows,env,ids):
    src=work/'src/coderl_lab/analysis';name='exp027_generated_'+weight[:16]
    code=(src/'exp021_seed42_generate.py').read_text()
    if code.count(OLD_SHA)!=1:raise ValueError('template model pin drift')
    code=code.replace(OLD_SHA,weight).replace('m.get("experiment")!="EXP-021"','m.get("experiment")!="EXP-024"')
    code=code.replace('EXP-021-7B-SFT-VS-GRPO-MBPPPLUS','EXP027-MATCHED-PAIRED96')
    target=src/(name+'.py')
    if target.exists() and target.read_text()!=code:raise ValueError('generated module drift')
    if not target.exists():target.write_text(code)
    run=out/('generate_'+weight[:16]);run.mkdir(parents=True,exist_ok=True)
    if not (run/'checkpoint.jsonl').exists():
        with (run/'checkpoint.jsonl').open('x') as f:
            for r in sft_rows:f.write(canonical({**r,'arm':'sft'})+'\n')
    if not (run/'generation_summary.json').exists():
        with open('/tmp/coderl_lab_gpu_experiment_lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            subprocess.run([sys.executable,'-m','coderl_lab.train.gpu_preflight','--min-free-mib','12000'],cwd=work,env=env,check=True)
            cmd=[sys.executable,'-m','coderl_lab.analysis.'+name,'--public',str(data_root/'public96_prompts.jsonl'),
                 '--manifest',str(data_root/'manifest.json'),'--sft','/home/you/projects/CodeRL-Lab-track-train011/artifacts/train011/qlora-ultrachat',
                 '--grpo',str(model),'--out',str(run)]
            with (run/'generate.log').open('a') as log:subprocess.run(cmd,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    result=from_bundle(run,'grpo',weight,ids)
    if result is None:raise ValueError('new generation identity absent')
    return result[0],{'kind':'new_generation','count':384,'source':str(run),'generator_sha256':ev.sha(target)}

def integration_gate(out):
    p=out/'evaluation_integration.json'
    if p.exists():return
    results={'correct':ev.execute('def f(x): return x+1','assert f(2)==3','f',cpu_seconds=1,wall_seconds=10),
             'budget':ev.execute('def f(x):\n while True: pass','f(1)','f',cpu_seconds=1,wall_seconds=10)}
    if not results['correct']['passed'] or results['budget']['reason']!='cpu_budget_exceeded':raise ValueError('real Docker integration failed')
    try:ev.execute('def f(x):\n import time; time.sleep(5)','f(1)','f',cpu_seconds=1,wall_seconds=1)
    except ev.SandboxIndeterminate:results['host_timeout_aborted']=True
    else:raise ValueError('host timeout scored')
    left=subprocess.run(['docker','ps','-aq','--filter','name=coderl-eval027-'],capture_output=True,text=True,check=True).stdout.strip()
    if left:raise ValueError('evaluation container residue')
    once(p,results)

def training_audit(root):
    r={};cache_observations={};source=None
    for seed in (43,44):
        traces={};stats={}
        for variant in ('legacy','corrected'):
            p=root/f'{variant}_seed{seed}';v=verify_run(p,variant,seed);stats[variant]=v
            s=json.loads((p/'summary.json').read_text())
            fingerprint=[s['train_sha256'],s['warm_start_source_sft_adapter_sha256'],s['active_training_loop_sha256'],s['active_reward_source_sha256']]
            if source is not None and source!=fingerprint:raise ValueError('training controls differ')
            source=fingerprint;traces[variant]=ev.rows(p/'reward_attempts.jsonl')
            for event in traces[variant]:
                key=event['probe_key']
                if key is None:continue
                if key in cache_observations and cache_observations[key]!=event['probe_sha256']:raise ValueError('same-code execution drift')
                cache_observations[key]=event['probe_sha256']
                record=json.loads((root/'observations'/(key+'.json')).read_text())
                if record['probe_sha256']!=event['probe_sha256'] or record['protocol']!=protocol():raise ValueError('training probe provenance drift')
        a=traces['legacy'];b=traces['corrected']
        if [x['task_id'] for x in a]!=[x['task_id'] for x in b]:raise ValueError('paired training task sequence changed')
        first=next((i for i,(x,y) in enumerate(zip(a,b)) if x['code_sha256']!=y['code_sha256']),None)
        r[str(seed)]={'arms':stats,'first_generated_code_difference':first,
                      'equal_model_file':stats['legacy']['adapter_sha256']==stats['corrected']['adapter_sha256'],
                      'same_code_observation_drift':0}
    return {'status':'complete_shared_execution_validated','seeds':r,'unique_observations':len(cache_observations),
            'conditioning':'fixed completed execution ledger; not a universal runtime-determinism claim'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--train-root',type=Path,required=True);p.add_argument('--eval-root',type=Path,required=True);a=p.parse_args()
    if os.uname().nodename!='Tang':raise RuntimeError('Tang required')
    work=Path(__file__).resolve().parents[1];root=a.train_root/'artifacts/exp027';out=root/'evaluation';out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(work/'src'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    deadline=time.monotonic()+5400
    while not (root/'four_training_runs.json').exists():
        if (root/'chain_failure.json').exists():raise RuntimeError('training failed; stop downstream')
        if time.monotonic()>deadline:raise TimeoutError('training summaries unavailable')
        time.sleep(10)
    audit=training_audit(root);once(root/'matched_training_audit.json',audit)
    integration_gate(out)
    data=a.eval_root/'artifacts/exp024/frozen_new96'
    if ev.sha(data/'public96_prompts.jsonl')!=ev.PUBLIC_SHA:raise ValueError('public cohort drift')
    public=ev.rows(data/'public96_prompts.jsonl');ids=sorted(x['task_id'] for x in public)
    if len(ids)!=96 or len(set(ids))!=96:raise ValueError('require 96 unique tasks')
    paths=[a.eval_root/f'artifacts/{part}' for part in ('exp024/eval_seed43_new96','exp024/eval_seed44_new96',
           'exp025/old_seed43_new96','exp025/old_seed44_new96','exp026_eval/depfix_seed43_new96','exp026_eval/depfix_seed44_new96')]
    baseline=from_bundle(paths[0],'sft',ev.SFT_SHA,ids)
    if baseline is None:raise ValueError('missing frozen SFT')
    selected={'sft':ev.SFT_SHA};models={ev.SFT_SHA:(baseline[0],baseline[1])}
    for seed in (43,44):
        for variant in ('legacy','corrected'):
            model=root/f'{variant}_seed{seed}';s=json.loads((model/'summary.json').read_text());weight=s['saved_adapter_sha256']
            selected[f'{variant}_{seed}']=weight
            if weight in models:continue
            found=next((z for path in paths if (z:=from_bundle(path,'grpo',weight,ids)) is not None),None)
            models[weight]=found if found else generate_new(work,data,model,weight,out,baseline[0],env,ids)
            print('GENERATION_READY',seed,variant,models[weight][1]['kind'],flush=True)
    sources={}
    for weight,(rr,origin) in models.items():
        gp=out/('generation_'+weight[:16]+'.jsonl');write_generation(gp,rr)
        sources[weight]={'file':str(gp),'sha256':ev.sha(gp),'count':len(rr),'provenance':origin}
    freeze={'status':'all_models_generated_frozen','models':selected,'sources':sources,'public_sha256':ev.PUBLIC_SHA,
            'new_candidates':sum(384 for rr,z in models.values() if z['kind']=='new_generation'),'eval_protocol':ev.protocol()}
    once(out/'generation_freeze.json',freeze)
    # First access to private test bytes: every model's answers are now frozen.
    if ev.sha(data/'private96_tests.jsonl')!=ev.PRIVATE_SHA:raise ValueError('private oracle drift')
    oracle={x['task_id']:x for x in ev.rows(data/'private96_tests.jsonl')}
    if set(oracle)!=set(ids):raise ValueError('oracle task mismatch')
    for weight,src in sources.items():
        if ev.sha(src['file'])!=src['sha256']:raise ValueError('frozen generation modified')
    golden_path=out/'golden.jsonl';gold={x['task_id']:x for x in ev.rows(golden_path)} if golden_path.exists() else {}
    for q in ids:
        if q not in gold:
            v=ev.score_cached(out/'score_cache',oracle[q]['canonical_code'],oracle[q]);r={'task_id':q,**v}
            with golden_path.open('a') as f:f.write(canonical(r)+'\n')
            gold[q]=r
    judged={}
    for weight,(rr,origin) in models.items():
        sp=out/('scored_'+weight[:16]+'.jsonl');done={(x['task_id'],x['sample_id']):x for x in ev.rows(sp)} if sp.exists() else {}
        for g in rr:
            k=(g['task_id'],g['sample_id'])
            if k in done:
                if done[k]['code_sha256']!=g['code_sha256']:raise ValueError('resumed score source mismatch')
                continue
            v=ev.score_cached(out/'score_cache',g['raw_code'],oracle[k[0]])
            r={'task_id':k[0],'sample_id':k[1],'code_sha256':g['code_sha256'],**v}
            with sp.open('a') as f:f.write(canonical(r)+'\n');f.flush()
            done[k]=r
            if len(done)%64==0:print('SCORED',weight[:12],len(done),'/384',flush=True)
        if len(done)!=384:raise ValueError('incomplete scores')
        judged[weight]=list(done.values())
    final={'experiment':'EXP027','status':'complete','protocol':ev.protocol(),'training_audit':audit,'generation':freeze,
           'golden_passed':sum(x['passed'] for x in gold.values()),'golden_total':96,
           'metrics':{name:ev.metrics(judged[w]) for name,w in selected.items()},'comparisons':{}}
    for seed in (43,44):
        final['comparisons'][str(seed)]=ev.compare(ev.matrix(judged[selected[f'legacy_{seed}']]),ev.matrix(judged[selected[f'corrected_{seed}']]))
    once(root/'matched_final.json',final)
    print('EXP027_MATCHED_COMPARISON_COMPLETE',flush=True)
if __name__=='__main__':main()
