"""Audit attribution validity separately from checkpoint/training completion.

Runtime drift is observed from the SAME candidate bytes and training task.
No unproven claim about CPU contention, hardware, or the ultimate cause is made.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path


def compare_trace(old:list[dict],new:list[dict])->dict:
    if not old or len(old)!=len(new):raise ValueError('trace length mismatch')
    if any(a['task_id']!=b['task_id'] for a,b in zip(old,new)):
        raise ValueError('training task order mismatch')
    if any('attempt_id' in r for r in new) and [r.get('attempt_id') for r in new]!=list(range(len(new))):
        raise ValueError('nonunique or unordered attempt IDs')
    drift=[];classification=[];code_changes=[]
    for i,(a,b) in enumerate(zip(old,new)):
        if 'legacy_reward_same_probe' not in b:raise ValueError('missing same-probe control')
        if b['reward']!=b['legacy_reward_same_probe']:
            classification.append({'attempt_id':i,'task_id':a['task_id'],
                'new_reward':b['reward'],'legacy_reward_same_probe':b['legacy_reward_same_probe']})
        if a['code_sha256']!=b['code_sha256']:
            code_changes.append(i);continue
        prior=(a['verdict'],a['executable_fraction'],a['dependency_fraction'],a['semantic_fraction'])
        current=(b['verdict'],b['executable_fraction'],b.get('legacy_dependency_fraction',b['dependency_fraction']),b['semantic_fraction'])
        if prior!=current:
            drift.append({'attempt_id':i,'task_id':a['task_id'],'code_sha256':a['code_sha256'],
                'old_verdict':a['verdict'],'new_verdict':b['verdict'],
                'old_reward':a['reward'],'new_legacy_reward':b['legacy_reward_same_probe'],
                'old_components':list(prior[1:]),'new_legacy_components':list(current[1:])})
    return {'status':'runtime_confounded' if drift else 'no_observed_same_code_execution_drift',
            'samples':len(old),'same_code_probe_drift':drift,
            'classification_effect_count':len(classification),'classification_effects':classification,
            'first_generation_code_difference':code_changes[0] if code_changes else None,
            'generated_code_difference_count':len(code_changes),
            'limitation':'Absence of observed drift is not a universal deterministic-execution guarantee.'}


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]

def main():
    p=argparse.ArgumentParser();p.add_argument('--train-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    report={'experiment':'EXP026-causal-validity','seeds':{},'sources':{},'source_sha256':digest(__file__)}
    for s in (43,44):
        old=a.train_root/f'artifacts/exp024/strict_seed{s}/reward_attempts.jsonl'
        new=a.train_root/f'artifacts/exp026/depfix_seed{s}/reward_attempts.jsonl'
        for path in (old,new):
            m=json.loads(path.with_name('summary.json').read_text())
            if m['reward_attempt_log_sha256']!=digest(path):raise ValueError('reward log mutation')
        report['seeds'][str(s)]=compare_trace(load(old),load(new))
        report['sources'][str(s)]={'old_reward_log_sha256':digest(old),'new_reward_log_sha256':digest(new)}
    report['status']='runtime_confounded' if any(x['status']=='runtime_confounded' for x in report['seeds'].values()) else 'no_observed_same_code_execution_drift'
    if a.out.exists():raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
