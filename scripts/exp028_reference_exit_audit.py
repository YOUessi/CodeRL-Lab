"""One bounded replay of the first unscored EXP028 reference control.

No model training/generation, no score replacement, no candidate inspection.
Same immutable image, driver, 1GiB/1CPU, 15/16 CPU budget and 60s wall guard.
The reference and hidden tests execute only inside the nonroot offline Docker.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,time,uuid
from pathlib import Path
from coderl_lab.analysis import exp027_matched_eval as ev
from coderl_lab.analysis.exp028_protocol import canonical,rows,sha,once


def validate_inputs(root):
    freeze=json.loads((root/'evaluation/generation_freeze.json').read_text())
    manifest=json.loads((root/'external96/manifest.json').read_text())
    if freeze.get('status')!='all_models_generated_frozen' or freeze['protocol']!=ev.protocol():
        raise ValueError('unfrozen generation or changed evaluation protocol')
    if freeze['public_sha256']!=manifest['public_sha256'] or sha(root/'external96/public.jsonl')!=manifest['public_sha256']:
        raise ValueError('public data drift')
    for weight,source in freeze['sources'].items():
        p=root/'evaluation'/('generation_'+weight[:16])
        summary=json.loads((p/'summary.json').read_text())
        if (source['count']!=384 or sha(p/'checkpoint.jsonl')!=source['generation_sha256']
            or summary['checkpoint_sha256']!=source['generation_sha256'] or summary['adapter_sha256']!=weight):
            raise ValueError('generation integrity check failed')
    public=rows(root/'external96/public.jsonl');ids={r['task_id'] for r in public}
    if len(public)!=96 or len(ids)!=96:raise ValueError('cohort size changed')
    done=rows(root/'evaluation/golden.jsonl')
    if len({r['task_id'] for r in done})!=len(done) or not {r['task_id'] for r in done}<=ids:
        raise ValueError('invalid reference checkpoint')
    pending=sorted(ids-{r['task_id'] for r in done})
    if not pending:raise ValueError('no pending reference')
    # All model generation checks are complete before reading hidden test bytes.
    private=root/'external96/private.jsonl'
    if sha(private)!=manifest['private_sha256']:raise ValueError('private data drift')
    oracle={r['task_id']:r for r in rows(private)}
    if set(oracle)!=ids:raise ValueError('reference cohort mismatch')
    return oracle[pending[0]],freeze,manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if os.uname().nodename!='Tang':raise RuntimeError('Tang required')
    if a.out.exists():raise FileExistsError('diagnostic exists; no unregistered retry')
    reference,freeze,manifest=validate_inputs(a.root)
    name='coderl-ref028-diagnostic-'+uuid.uuid4().hex[:16]
    record={'experiment':'EXP028-reference-exit-diagnostic','task_id':reference['task_id'],
        'reference_code_sha256':hashlib.sha256(reference['canonical_code'].encode()).hexdigest(),
        'test_sha256':hashlib.sha256(reference['test'].encode()).hexdigest(),
        'generation_freeze_sha256':sha(a.root/'evaluation/generation_freeze.json'),
        'public_sha256':manifest['public_sha256'],'protocol':ev.protocol(),'memory_bytes':1024**3,
        'diagnostic_only':True,'historical_verdicts_changed':False,'model_candidates_opened':False}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    once(a.out.with_suffix('.prepared.json'),record)
    cmd=['docker','run','--name',name,'-i','--network','none','--memory','1g','--cpus','1','--pids-limit','64',
         '--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--user','65534:65534',
         '--tmpfs','/tmp:rw,nosuid,size=128m','-e','PYTHONDONTWRITEBYTECODE=1','-e','OPENBLAS_NUM_THREADS=1',
         '--entrypoint','/usr/bin/python3',ev.IMAGE_ID,'-c',ev.DRIVER]
    start=time.monotonic()
    try:
        try:
            result=subprocess.run(cmd,input=json.dumps({'code':reference['canonical_code'],'test':reference['test'],
                'entry_point':reference['entry_point']}),capture_output=True,text=True,timeout=60)
            record['exit_code']=result.returncode
            record['cpu_budget_marker_count']=result.stdout.splitlines().count('__EXP027_CPU_BUDGET__')
            record['semantic_verdict_count']=sum(x.startswith('__EXP019B__') for x in result.stdout.splitlines())
        except subprocess.TimeoutExpired:
            record['host_timeout']=True
        state=subprocess.run(['docker','inspect','--format','{{json .State}}',name],capture_output=True,text=True,check=True,timeout=10)
        d=json.loads(state.stdout)
        record['container_state']={k:d.get(k) for k in ('Running','OOMKilled','Dead','ExitCode','Error')}
    finally:
        clean=subprocess.run(['docker','rm','-f',name],capture_output=True,text=True,timeout=15)
        if clean.returncode and 'No such container' not in clean.stderr:
            raise RuntimeError('cleanup not confirmed')
    record['cleanup_confirmed']=True;record['elapsed_seconds']=round(time.monotonic()-start,4)
    once(a.out,record);print(canonical(record),flush=True)
if __name__=='__main__':main()
