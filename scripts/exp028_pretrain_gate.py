"""Complete bounded pretraining gate. No external hidden test bytes are read."""
from __future__ import annotations
import argparse,fcntl,json,os,subprocess,sys
from pathlib import Path
from coderl_lab.analysis.exp028_protocol import (SFT_SHA,canonical,sha,rows,once,gate,validate_generations,public_rows)
from coderl_lab.train.exp027_matched_reward import observe,rewards
from coderl_lab.train.train014b_reward_sandbox import normalize_code,check_code

SFT=Path('/home/you/projects/CodeRL-Lab-track-train011/artifacts/train011/qlora-ultrachat')

def check_data(root):
    data=root/'data';m=json.loads((data/'manifest.json').read_text())
    for rel,h in m['files'].items():
        if sha(data/rel)!=h:raise ValueError('frozen training file changed: '+rel)
    return data,m


def main_run(root,work):
    if os.uname().nodename!='Tang':raise RuntimeError('Tang required')
    data,m=check_data(root)
    tasks={x['task_id']:x for x in rows(data/'gate_train_records.jsonl')}
    if len(tasks)!=72:raise ValueError('unique task count')
    p=root/'reference_checks.jsonl';checked={x['task_id']:x for x in rows(p)} if p.exists() else {}
    for reference in rows(data/'references.jsonl'):
        q=reference['task_id'];code=reference['code'];t=tasks[q]['test_cases']
        if q in checked:
            if checked[q]['source_sha256']!=__import__('hashlib').sha256(code.encode()).hexdigest():raise ValueError('reference drift')
            continue
        obs=observe(root/'reference_observations',code,t);r=rewards(obs['probe'])
        if r['semantic_fraction']!=1 or r['executable_fraction']!=1:raise RuntimeError('reference solution failed '+q)
        record={'task_id':q,'source_sha256':__import__('hashlib').sha256(code.encode()).hexdigest(),'probe_key':obs['key'],'reward':r['corrected_reward']}
        with p.open('a') as f:f.write(canonical(record)+'\n');f.flush()
        checked[q]=record
    if set(checked)!=set(tasks):raise ValueError('reference cohort mismatch')
    once(root/'references_complete.json',{'count':72,'checks_sha256':sha(p),'data_manifest_sha256':sha(data/'manifest.json')})
    print('REFERENCE_GATE_PASSED',len(checked),flush=True)
    generated=root/'pretrain_generation'
    if not (generated/'summary.json').exists():
        env=dict(os.environ,PYTHONPATH=str(work/'src'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
        with open('/tmp/coderl_lab_gpu_experiment_lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            subprocess.run([sys.executable,'-m','coderl_lab.train.gpu_preflight','--min-free-mib','12000'],env=env,cwd=work,check=True)
            with (root/'pretrain_generate.log').open('a') as log:
                subprocess.run([sys.executable,'-m','coderl_lab.analysis.exp028_generate','--public',str(data/'gate_public.jsonl'),
                                '--public-sha',sha(data/'gate_public.jsonl'),'--adapter',str(SFT),'--adapter-sha',SFT_SHA,'--out',str(generated)],
                               cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    s=json.loads((generated/'summary.json').read_text())
    if s['checkpoint_sha256']!=sha(generated/'checkpoint.jsonl') or s['adapter_sha256']!=SFT_SHA:raise ValueError('generation not frozen')
    gr,expected=validate_generations(rows(generated/'checkpoint.jsonl'),public_rows(data/'gate_public.jsonl'))
    if set(gr)!=expected:raise ValueError('not all gate generations complete')
    scored=root/'pretrain_scored.jsonl';done={(x['task_id'],x['sample_id']):x for x in rows(scored)} if scored.exists() else {}
    for key,g in sorted(gr.items()):
        if key in done:
            if done[key]['code_sha256']!=g['code_sha256']:raise ValueError('score source changed')
            continue
        code,norm=normalize_code(g['raw_code']);valid,entry=check_code(code)
        obs=None
        if valid and entry:
            obs=observe(root/'training_observations',code,tasks[key[0]]['test_cases']);f=rewards(obs['probe'])
        else:f={'corrected_reward':0.,'semantic_fraction':0.,'verdict':'invalid_code'}
        result={'task_id':key[0],'sample_id':key[1],'code_sha256':g['code_sha256'],'reward':f['corrected_reward'],
                'syntax_valid':bool(valid and entry),'clipped':bool(g['reached_token_limit']),'semantic_fraction':f['semantic_fraction'],
                'is_new':key[0].startswith('train/coverage/'),'probe_key':obs['key'] if obs else None}
        with scored.open('a') as file:file.write(canonical(result)+'\n');file.flush()
        done[key]=result
        if len(done)%32==0:print('GATE_SCORED',len(done),'/288',flush=True)
    if set(done)!=expected:raise ValueError('gate scoring incomplete')
    by_arm={}
    for arm in ('narrow','broad'):
        ids={x['task_id'] for x in rows(data/arm/'train.jsonl')}
        by_arm[arm]=[r for k,r in done.items() if k[0] in ids]
    report=gate(by_arm)
    report.update({'status':'gate_passed' if report['passed'] else 'stopped_by_preregistered_gate',
                   'unique_generated_candidates':288,'logical_candidates_per_arm':192,'shared_candidates':96,
                   'generation_sha256':sha(generated/'checkpoint.jsonl'),'score_sha256':sha(scored),
                   'data_manifest_sha256':sha(data/'manifest.json'),'external_hidden_tests_opened':False})
    once(root/'gate_report.json',report)
    print('GATE_FINAL',canonical(report),flush=True)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();work=Path(__file__).resolve().parents[1]
    try:main_run(a.root,work)
    except Exception as e:
        once(a.root/'gate_failure.json',{'error':type(e).__name__,'reason':str(e),'training_started':False})
        raise
if __name__=='__main__':main()
