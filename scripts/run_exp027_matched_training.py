"""Four bounded matched-protocol runs on Tang, fail closed on any missing evidence."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path
import yaml
from coderl_lab.train.exp027_matched_reward import protocol,file_sha,canonical


def verify_run(path,variant,seed):
    s=json.loads((path/'summary.json').read_text())
    if (s['status']!='real_GPU_GRPO_completed' or s['global_step']!=32 or
        s['training_seed']!=seed or s['matched_variant']!=variant or s['execution_protocol']!=protocol() or
        s['saved_adapter_sha256']!=file_sha(path/'adapter_model.safetensors') or
        s['reward_attempt_log_sha256']!=file_sha(path/'reward_attempts.jsonl')):
        raise ValueError('training evidence incomplete/drifted')
    rows=[json.loads(x) for x in (path/'reward_attempts.jsonl').read_text().splitlines()]
    if len(rows)!=128 or any(r['variant']!=variant for r in rows):raise ValueError('wrong sample budget')
    if (path/'unscorable_abort.json').exists():raise ValueError('unscorable run cannot be completed')
    return {'variant':variant,'seed':seed,'adapter_sha256':s['saved_adapter_sha256'],
            'optimizer_steps':32,'reward_samples':128,'cache_hits':sum(r['cache_hit'] for r in rows),
            'docker_executions':sum(r['probe_key'] is not None and not r['cache_hit'] for r in rows),
            'changed_elements_historical_compatible':s['initial_SFT_parameter_elements_changed_after_grpo'],
            'classification_effect_samples':sum(r['legacy_reward']!=r['corrected_reward'] for r in rows)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--train-root',type=Path,required=True);a=p.parse_args()
    if os.uname().nodename!='Tang':raise RuntimeError('Tang required')
    work=Path(__file__).resolve().parents[1];out=a.train_root/'artifacts/exp027';out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(work/'src'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONHASHSEED='0')
    identity={'experiment':'EXP027','protocol':protocol(),'training_code_sha':file_sha(work/'src/coderl_lab/train/exp027_7b_grpo.py'),
              'reward_code_sha':file_sha(work/'src/coderl_lab/train/exp027_matched_reward.py'),'order':[[s,v] for s in (43,44) for v in ('legacy','corrected')]}
    ip=out/'chain_identity.json'
    if ip.exists() and json.loads(ip.read_text())!=identity:raise ValueError('chain identity changed')
    if not ip.exists():ip.write_text(canonical(identity)+'\n')
    with (out/'chain.lock').open('a') as chain:
        fcntl.flock(chain,fcntl.LOCK_EX|fcntl.LOCK_NB)
        completed=[]
        for seed in (43,44):
            for variant in ('legacy','corrected'):
                run=out/f'{variant}_seed{seed}'
                cfg=yaml.safe_load((a.train_root/f'configs/exp024_strict_seed{seed}.yaml').read_text())
                cfg['experiment']['matched_variant']=variant
                cfg['reward']['wall_guard_seconds']=60.
                cfg['reward']['timeout_seconds']=60.
                cfg['runs']['formal_output']=str(run)
                cfg['runs']['smoke_output']=str(out/f'smoke_{variant}_{seed}')
                cp=out/f'{variant}_seed{seed}.yaml';text=yaml.safe_dump(cfg,sort_keys=False)
                if cp.exists() and cp.read_text()!=text:raise ValueError('config drift')
                if not cp.exists():cp.write_text(text)
                if not (run/'summary.json').is_file():
                    if run.exists():raise RuntimeError('partial training exists; do not silently restart')
                    print('START',seed,variant,flush=True)
                    with open('/tmp/coderl_lab_gpu_experiment_lock','a') as lock:
                        fcntl.flock(lock,fcntl.LOCK_EX)
                        subprocess.run([sys.executable,'-m','coderl_lab.train.gpu_preflight','--min-free-mib','12000'],cwd=work,env=env,check=True)
                        cmd=[sys.executable,'-m','coderl_lab.train.exp027_7b_grpo','--variant',variant,'--cache-root',str(out/'observations'),
                             '--arm','strict','--config',str(cp),'--data-dir',str(a.train_root/'artifacts/train017/frozen24'),
                             '--sft-adapter','/home/you/projects/CodeRL-Lab-track-train011/artifacts/train011/qlora-ultrachat',
                             '--output-dir',str(run),'--max-steps','32','--run-kind','formal']
                        with (out/f'{variant}_seed{seed}.log').open('a') as log:
                            r=subprocess.run(cmd,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT)
                        if r.returncode:
                            (out/'chain_failure.json').write_text(canonical({'seed':seed,'variant':variant,'exit':r.returncode})+'\n')
                            raise RuntimeError('training stopped; reward/evidence failure not ignored')
                s=verify_run(run,variant,seed);completed.append(s)
                (out/'training_progress.json').write_text(canonical({'completed':completed})+'\n')
                print('VERIFIED',canonical(s),flush=True)
        (out/'four_training_runs.json').write_text(canonical({'status':'completed','runs':completed,'identity':identity})+'\n')
        print('EXP027_FOUR_TRAININGS_COMPLETE',flush=True)
if __name__=='__main__':main()
