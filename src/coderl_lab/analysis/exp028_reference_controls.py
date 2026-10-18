"""Bounded reference-control audit; never evaluate or rank model candidates.

This adds verified engine OOM classification to the already fixed evaluation
budget. Original failures and scores are immutable. A failed reference prevents
model-scoring claims; no task is dropped and no budget is increased.
"""
from __future__ import annotations
import argparse,collections,datetime,hashlib,json,os,subprocess,time,uuid
from pathlib import Path
from coderl_lab.analysis import exp027_matched_eval as ev
from coderl_lab.analysis.exp028_protocol import canonical,rows,sha,once
from coderl_lab.train.exp027_guarded_probe import SandboxIndeterminate


def protocol():
    return {**ev.protocol(),'reference_audit':'EXP028-reference-engine-evidence-v1',
            'memory_bytes':1024**3,'cpu_hard_seconds':16,
            'oom_classification':'engine_OOMKilled_and_stopped_and_exit137',
            'code_sha256':sha(__file__),'budgets_changed':False}


def classify_exit(rc,stdout,state):
    if (type(rc) is not int or state.get('Running') is not False
        or type(state.get('ExitCode')) is not int or state['ExitCode']!=rc):
        raise SandboxIndeterminate('reference_engine_exit_not_confirmed')
    if rc==137 and state.get('OOMKilled') is True:
        return {'passed':False,'reason':'memory_budget_exceeded','resource_limited':True}
    if state.get('OOMKilled') is not False:
        raise SandboxIndeterminate('reference_engine_memory_state_unknown')
    return ev.decode_exit(rc,stdout)


def control_summary(records,ids):
    ids=set(ids);found={}
    for r in records:
        q=r['task_id']
        if q in found or q not in ids:raise ValueError('reference task mismatch or duplicate')
        if type(r.get('passed')) is not bool or not isinstance(r.get('reason'),str):
            raise ValueError('invalid reference verdict')
        if r['passed']!=(r['reason']=='none'):raise ValueError('reference verdict contradiction')
        found[q]=r
    if not ids or set(found)!=ids:raise ValueError('incomplete reference controls')
    failures={q:r['reason'] for q,r in found.items() if not r['passed']}
    return {'status':'stopped_reference_controls' if failures else 'reference_controls_passed',
            'passed':len(ids)-len(failures),'failed':len(failures),'total':len(ids),'dropped_tasks':0,
            'failure_types':dict(collections.Counter(failures.values())),'failed_task_ids':sorted(failures),
            'model_scoring_permitted':not failures}


def execute_reference(oracle,out_dir):
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    code=oracle['canonical_code'];test=oracle['test'];entry=oracle['entry_point']
    fingerprint={'task_id':oracle['task_id'],'reference_code_sha256':hashlib.sha256(code.encode()).hexdigest(),
                 'test_sha256':hashlib.sha256(test.encode()).hexdigest(),'entry_point':entry,'protocol':protocol()}
    key=ev.objsha(fingerprint);prefix=out_dir/key;complete=prefix.with_suffix('.complete.json')
    if complete.exists():
        saved=json.loads(complete.read_text());body={k:v for k,v in saved.items() if k!='record_sha256'}
        if saved.get('record_sha256')!=ev.objsha(body) or any(saved.get(k)!=v for k,v in fingerprint.items()):
            raise ValueError('reference evidence changed')
        if saved.get('cleanup_confirmed') is not True:raise ValueError('unconfirmed cleanup in reference record')
        return saved
    prepared=prefix.with_suffix('.prepared.json')
    if prepared.exists():raise RuntimeError('uncommitted reference attempt; no automatic retry')
    start=time.monotonic();name='coderl-ref028-control-'+uuid.uuid4().hex[:16]
    record={**fingerprint,'key':key,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'candidate_scoring':False}
    once(prepared,record)
    cmd=['docker','run','--name',name,'-i','--network','none','--memory','1g','--cpus','1','--pids-limit','64',
         '--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--user','65534:65534',
         '--tmpfs','/tmp:rw,nosuid,size=128m','-e','PYTHONDONTWRITEBYTECODE=1','-e','OPENBLAS_NUM_THREADS=1',
         '--entrypoint','/usr/bin/python3',ev.IMAGE_ID,'-c',ev.DRIVER]
    error=None;verdict=None
    try:
        try:
            r=subprocess.run(cmd,input=json.dumps({'code':code,'test':test,'entry_point':entry}),
                             capture_output=True,text=True,timeout=60)
        except subprocess.TimeoutExpired as exc:
            raise SandboxIndeterminate('reference_wall_guard_no_verdict') from exc
        p=subprocess.run(['docker','inspect','--format','{{json .State}}',name],
                         capture_output=True,text=True,check=True,timeout=10)
        s=json.loads(p.stdout)
        record.update({'exit_code':r.returncode,'container_state':{k:s.get(k) for k in ('Running','OOMKilled','ExitCode','Dead','Error')},
                       'cpu_budget_marker_count':r.stdout.splitlines().count('__EXP027_CPU_BUDGET__')})
        once(prefix.with_suffix('.engine.json'),record)
        verdict=classify_exit(r.returncode,r.stdout,s)
    except Exception as exc:
        error=exc
    finally:
        try:
            c=subprocess.run(['docker','rm','-f',name],capture_output=True,text=True,timeout=15)
            if c.returncode and 'No such container' not in c.stderr:
                raise SandboxIndeterminate('reference_container_cleanup_failed')
            record['cleanup_confirmed']=True
        except (OSError,subprocess.SubprocessError,SandboxIndeterminate) as exc:
            error=SandboxIndeterminate('reference_container_cleanup_unconfirmed')
            record['cleanup_error_type']=type(exc).__name__
    record['elapsed_seconds']=round(time.monotonic()-start,4)
    if error is not None:
        once(prefix.with_suffix('.failed.json'),{**record,'error_type':type(error).__name__,'reason':str(error),
                                               'reward_or_score_assigned':False})
        raise error
    record.update(verdict);record['record_sha256']=ev.objsha(record)
    once(complete,record)
    return record


def validate_frozen_run(root):
    freeze_path=root/'evaluation/generation_freeze.json'
    freeze=json.loads(freeze_path.read_text());m=json.loads((root/'external96/manifest.json').read_text())
    if freeze.get('status')!='all_models_generated_frozen' or freeze['protocol']!=ev.protocol():
        raise ValueError('generation freeze/protocol mismatch')
    if (sha(root/'external96/public.jsonl')!=m['public_sha256'] or freeze['public_sha256']!=m['public_sha256']
        or m['selected']!=96 or m['excluded_task_count']!=216):raise ValueError('external cohort changed')
    public=rows(root/'external96/public.jsonl');ids={r['task_id'] for r in public}
    if len(public)!=96 or len(ids)!=96:raise ValueError('need exactly 96 unique tasks')
    expected_names={'sft','narrow_43','narrow_44','broad_43','broad_44'}
    if set(freeze['models'])!=expected_names:raise ValueError('model cohort mismatch')
    data=json.loads((root/'data/manifest.json').read_text())
    for rel,h in data['files'].items():
        if sha(root/'data'/rel)!=h:raise ValueError('training data changed')
    proofs=json.loads((root/'four_trainings.json').read_text())
    if len(proofs)!=4 or {(x['arm'],x['seed']) for x in proofs}!={(a,s) for a in ('narrow','broad') for s in (43,44)}:
        raise ValueError('incomplete matched trainings')
    for p in proofs:
        name=f"{p['arm']}_seed{p['seed']}";d=root/name;s=json.loads((d/'summary.json').read_text())
        if (s['status']!='real_GPU_GRPO_completed' or s['global_step']!=32
            or s['saved_adapter_sha256']!=sha(d/'adapter_model.safetensors')
            or s['saved_adapter_sha256']!=p['adapter_sha256']
            or freeze['models'][f"{p['arm']}_{p['seed']}"]!=p['adapter_sha256']
            or s['warm_start_source_sft_adapter_sha256']!=ev.SFT_SHA
            or s['train_sha256']!=sha(root/'data'/p['arm']/'train.jsonl')
            or s['reward_attempt_log_sha256']!=sha(d/'reward_attempts.jsonl')
            or len(rows(d/'reward_attempts.jsonl'))!=128 or (d/'unscorable_abort.json').exists()):
            raise ValueError('training proof drift')
        if len(list((d/'pending_groups').glob('group_*.json')))!=32 or len(list((d/'reward_groups_ready').glob('group_*.json')))!=32:
            raise ValueError('durable training groups incomplete')
    for weight,source in freeze['sources'].items():
        p=root/'evaluation'/('generation_'+weight[:16]);s=json.loads((p/'summary.json').read_text())
        if (source['count']!=384 or sha(p/'checkpoint.jsonl')!=source['generation_sha256']
            or s['checkpoint_sha256']!=source['generation_sha256'] or s['adapter_sha256']!=weight
            or s['public_sha256']!=m['public_sha256']):raise ValueError('generation proof drift')
        rr=rows(p/'checkpoint.jsonl');keys={(x['task_id'],x['sample_id']) for x in rr}
        if len(rr)!=384 or keys!={(q,i) for q in ids for i in range(4)}:raise ValueError('incomplete generation')
        if any(hashlib.sha256(x['raw_code'].encode()).hexdigest()!=x['code_sha256'] for x in rr):
            raise ValueError('raw generation hash drift')
    # No private bytes read until all model outputs and training evidence validate.
    private=root/'external96/private.jsonl'
    if sha(private)!=m['private_sha256']:raise ValueError('hidden reference data changed')
    oracle={x['task_id']:x for x in rows(private)}
    if set(oracle)!=ids:raise ValueError('private reference task mismatch')
    return ids,oracle,{'generation_freeze_sha256':sha(freeze_path),'four_trainings_sha256':sha(root/'four_trainings.json'),
        'public_sha256':m['public_sha256'],'private_sha256':m['private_sha256'],'training_runs':proofs,
        'frozen_generation_records':sum(x['count'] for x in freeze['sources'].values())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if os.uname().nodename!='Tang':raise RuntimeError('Tang only')
    a.out.mkdir(parents=True,exist_ok=True)
    ids,oracle,source=validate_frozen_run(a.root)
    plan={'experiment':'EXP028-reference-controls','source':source,'protocol':protocol(),
          'model_candidates_scored':0,'reference_failures_stop_model_comparison':True,'tasks_dropped':0}
    once(a.out/'audit_identity.json',plan)
    records=[]
    try:
        for q in sorted(ids):
            record=execute_reference(oracle[q],a.out/'evidence');records.append(record)
            print('REFERENCE',len(records),'/96',q,record['reason'],flush=True)
        result={**plan,**control_summary(records,ids),'reference_records_sha256':ev.objsha(records),
                'per_task':[{'task_id':x['task_id'],'passed':x['passed'],'reason':x['reason'],
                             'resource_limited':x['resource_limited'],'record_sha256':x['record_sha256']} for x in records]}
        once(a.out/'final.json',result)
        print('REFERENCE_CONTROLS_FINAL',canonical({k:result[k] for k in ('status','passed','failed','total','failure_types','model_scoring_permitted')}),flush=True)
    except Exception as exc:
        once(a.out/'audit_failure.json',{'error_type':type(exc).__name__,'reason':str(exc),'completed_controls':len(records),
                                       'model_candidates_scored':0,'no_automatic_retry':True})
        raise
if __name__=='__main__':main()
