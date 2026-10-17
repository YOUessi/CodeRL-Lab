"""One bounded diagnostic of the first unscored frozen training candidate.
Same CPU/memory/no-network restrictions. Does not modify gate or award rewards.
Retains the container only long enough to inspect State; finally removes it.
"""
import argparse,hashlib,json,subprocess,time,uuid
from pathlib import Path
from coderl_lab.analysis.exp028_protocol import rows,sha,canonical,once
from coderl_lab.train.train014b_reward_sandbox import normalize_code,IMAGE_ID
from coderl_lab.train.exp024_reward import DRIVER


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    if (root/'exit_diagnostic.json').exists():raise FileExistsError('diagnostic already exists')
    summary=json.loads((root/'pretrain_generation/summary.json').read_text())
    if summary['checkpoint_sha256']!=sha(root/'pretrain_generation/checkpoint.jsonl'):raise ValueError('generation changed')
    done={(r['task_id'],r['sample_id']) for r in rows(root/'pretrain_scored.jsonl')}
    generated={(r['task_id'],r['sample_id']):r for r in rows(root/'pretrain_generation/checkpoint.jsonl')}
    key=next(k for k in sorted(generated) if k not in done)
    task=next(r for r in rows(root/'data/gate_train_records.jsonl') if r['task_id']==key[0])
    code,_=normalize_code(generated[key]['raw_code']);name='coderl-diag028-'+uuid.uuid4().hex[:16]
    cmd=['docker','run','--name',name,'-i','--network','none','--read-only','--cap-drop','ALL',
         '--security-opt','no-new-privileges','--user','65534:65534','--memory','256m','--cpus','1','--pids-limit','32',
         '--tmpfs','/tmp:rw,nosuid,size=16m','--workdir','/tmp','-e','PYTHONDONTWRITEBYTECODE=1','--entrypoint','python',IMAGE_ID,'-I','-c',DRIVER]
    result={'task_id':key[0],'sample_id':key[1],'raw_code_sha256':generated[key]['code_sha256'],
            'driver_sha256':hashlib.sha256(DRIVER.encode()).hexdigest(),'image_id':IMAGE_ID,'memory_bytes':256*1024*1024,
            'diagnostic_only':True,'reward_assigned':False}
    started=time.monotonic()
    try:
        try:
            r=subprocess.run(cmd,input=json.dumps({'code':code,'tests':task['test_cases']}),capture_output=True,text=True,timeout=60)
            result['exit_code']=r.returncode;result['stdout_verdict_present']='__EXP024__' in r.stdout
        except subprocess.TimeoutExpired:result['host_timeout']=True
        state=subprocess.run(['docker','inspect','--format','{{json .State}}',name],capture_output=True,text=True,check=True,timeout=10)
        d=json.loads(state.stdout);result['container_state']={k:d.get(k) for k in ('Running','OOMKilled','Dead','ExitCode','Error')}
    finally:
        cleanup=subprocess.run(['docker','rm','-f',name],capture_output=True,text=True,timeout=15)
        if cleanup.returncode and 'No such container' not in cleanup.stderr:raise RuntimeError('cleanup failed')
    result['runtime_seconds']=round(time.monotonic()-started,3);result['cleanup_confirmed']=True
    once(root/'exit_diagnostic.json',result);print(canonical(result),flush=True)
if __name__=='__main__':main()
