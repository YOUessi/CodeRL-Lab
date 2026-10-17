"""Continue EXP026 through full same-cohort evaluation without manual handoffs.

Only the corrected adapter generates new answers: 384 per seed. The other 384
SFT records are reused from a validated frozen EXP024 checkpoint. This is paired
re-evaluation of an already observed cohort, not a new blind benchmark.
"""
from __future__ import annotations
import argparse,hashlib,json,os,shlex,subprocess,sys,time
from pathlib import Path
from coderl_lab.analysis.exp026_runner import load_bundle,rows,sha,write_once,PUBLIC_SHA

OLD_SHA='47e0d8d6687efe1965c2ba7ff88f064b56fe3d10907c742824b116abe1ffc8b4'
SFT_SHA='25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8'

def prepare(work:Path,ev:Path,train:Path,seed:int):
    if seed not in (43,44):raise ValueError('unregistered training seed')
    checkpoint=train/f'artifacts/exp026/depfix_seed{seed}'
    s=json.loads((checkpoint/'summary.json').read_text())
    weight_sha=sha(checkpoint/'adapter_model.safetensors')
    if (s['status']!='real_GPU_GRPO_completed' or s['global_step']!=32
        or s['saved_adapter_sha256']!=weight_sha or s['warm_start_source_sft_adapter_sha256']!=SFT_SHA):
        raise ValueError('unverified trained adapter')
    public=ev/'artifacts/exp024/frozen_new96/public96_prompts.jsonl'
    if sha(public)!=PUBLIC_SHA:raise ValueError('cohort drift')
    ids={r['task_id'] for r in rows(public)}
    source=ev/f'artifacts/exp024/eval_seed{seed}_new96'
    g,_,origin=load_bundle(source,ids)
    src=work/'src/coderl_lab/analysis'
    gen=(src/'exp021_seed42_generate.py').read_text()
    scorer=(src/'exp021_seed42_docker_score.py').read_text()
    if gen.count(OLD_SHA)!=1:raise ValueError('generator template drift')
    gen=gen.replace(OLD_SHA,weight_sha).replace('m.get("experiment")!="EXP-021"','m.get("experiment")!="EXP-024"')
    gen=gen.replace('EXP-021-7B-SFT-VS-GRPO-MBPPPLUS','EXP026-DEPENDENCY-FIX-PAIRED96')
    marker='"answers_generated":len(completed),'
    if gen.count(marker)!=1:raise ValueError('generation count marker changed')
    gen=gen.replace(marker,marker+'\n        "reused_sft_generations":384,\n        "new_model_generations":len(completed)-384,')
    name=f'exp026_eval_seed{seed}'
    old_import='from coderl_lab.analysis.exp021_seed42_generate import'
    if scorer.count(old_import)!=1:raise ValueError('scorer template drift')
    scorer=scorer.replace(old_import,f'from coderl_lab.analysis.{name}_generate import')
    for suffix,content in [('generate',gen),('docker_score',scorer)]:
        p=src/f'{name}_{suffix}.py'
        if p.exists() and p.read_text()!=content:raise ValueError('refuse generated-source drift')
        if not p.exists():p.write_text(content)
    out=ev/f'artifacts/exp026_eval/depfix_seed{seed}_new96'
    out.mkdir(parents=True,exist_ok=True)
    provenance={'experiment':'EXP026-paired-eval','seed':seed,'new_adapter_sha256':weight_sha,
                'sft_source_checkpoint':origin,'public_sha256':PUBLIC_SHA,'reused_sft_count':384,
                'generator_sha256':sha(src/f'{name}_generate.py'),'scorer_sha256':sha(src/f'{name}_docker_score.py')}
    marker_file=out/'reuse_provenance.json'
    if marker_file.exists() and json.loads(marker_file.read_text())!=provenance:raise ValueError('reuse provenance changed')
    if not marker_file.exists():write_once(marker_file,provenance)
    cp=out/'checkpoint.jsonl'
    if not cp.exists():
        with cp.open('x') as f:
            for q in sorted(ids):
                for i in range(4):f.write(json.dumps(g[q,i,'sft'],ensure_ascii=False)+'\n')
    else:
        reused={(r['task_id'],r['sample_id']):r for r in rows(cp) if r['arm']=='sft'}
        if len(reused)!=384 or any(reused[q,i]!=g[q,i,'sft'] for q in ids for i in range(4)):
            raise ValueError('SFT reuse drift')
    return checkpoint,out,name

def main():
    p=argparse.ArgumentParser();p.add_argument('--train-root',type=Path,required=True);p.add_argument('--eval-root',type=Path,required=True);a=p.parse_args()
    work=Path(__file__).resolve().parents[1]
    env=dict(os.environ,PYTHONPATH=str(work/'src'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    data=a.eval_root/'artifacts/exp024/frozen_new96'
    for seed in (43,44):
        deadline=time.monotonic()+1800
        while not (a.train_root/f'artifacts/exp026/depfix_seed{seed}/summary.json').is_file():
            if time.monotonic()>deadline:raise TimeoutError('training did not commit summary')
            time.sleep(5)
        model,out,name=prepare(work,a.eval_root,a.train_root,seed)
        if not (out/'generation_summary.json').exists():
            preflight=[sys.executable,'-m','coderl_lab.train.gpu_preflight','--min-free-mib','12000']
            generate=[sys.executable,'-m',f'coderl_lab.analysis.{name}_generate','--public',str(data/'public96_prompts.jsonl'),
                      '--manifest',str(data/'manifest.json'),'--sft','/home/you/projects/CodeRL-Lab-track-train011/artifacts/train011/qlora-ultrachat',
                      '--grpo',str(model),'--out',str(out)]
            with (out/'generate.log').open('a') as log:
                subprocess.run(['flock','-x','/tmp/coderl_lab_gpu_experiment_lock','bash','-c',shlex.join(preflight)+' && '+shlex.join(generate)],cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        frozen=json.loads((out/'generation_summary.json').read_text())
        if frozen['answers_generated']!=768 or frozen['new_model_generations']!=384 or frozen['reused_sft_generations']!=384:
            raise ValueError('incomplete or mislabelled generation')
        if not (out/'score_summary.json').exists():
            cmd=[sys.executable,'-m',f'coderl_lab.analysis.{name}_docker_score','--public',str(data/'public96_prompts.jsonl'),
                 '--private',str(data/'private96_tests.jsonl'),'--manifest',str(data/'manifest.json'),'--generation-dir',str(out),'--workers','2']
            with (out/'score.log').open('a') as log:subprocess.run(cmd,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('EXP026_PAIRED_EVAL_COMPLETE',seed,flush=True)
    print('EXP026_ALL_EVALUATIONS_COMPLETE',flush=True)
if __name__=='__main__':main()
