"""Diagnostic replay of frozen train candidates 22/23; no benchmark tests.

Protocol A reproduces the 8s host wall guard. Protocol B measures an explicit
8s CPU budget with a separate 30s safety wall guard. These are DIFFERENT
protocols, not permission to change historical rewards. Every container is
removed even when the host guard times out. No raw code or tests are exported.
"""
from __future__ import annotations
import argparse,hashlib,json,subprocess,time,uuid
from pathlib import Path
from coderl_lab.train.train014b_reward_sandbox import validate_tests
IMAGE='sha256:d3846bce6e97d0ef5c32fc023300d5e39251ad791e6fdfaee41d2ba75246766e'
TRAIN_SHA='21a63fcef2e74d7e902480db9aabceb6b61d5f3bd4aff551e1959f7b8e38e4bb'
DRIVER=r'''
import contextlib,io,json,os,resource,signal,sys,time
p=json.load(sys.stdin);cpu0=time.process_time();wall0=time.monotonic();done=0;passed=0
emit=lambda x: print('__EXP026_TIME__'+json.dumps(x),flush=True)
def stop(signum,frame):
 emit({'status':'cpu_budget_exhausted','completed_cases':done,'passed':passed,'cpu_seconds':time.process_time()-cpu0,'wall_seconds':time.monotonic()-wall0})
 os._exit(124)
if p['cpu_limit']:
 signal.signal(signal.SIGXCPU,stop)
 resource.setrlimit(resource.RLIMIT_CPU,(8,9))
emit({'status':'started'})
space={'__name__':'__candidate__'}
try:
 with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
  exec(compile(p['code'],'<candidate>','exec'),space,space)
  fn=space['solve']
except BaseException as ex:
 emit({'status':'load_error','error':type(ex).__name__});sys.exit(0)
errors={}
for t in p['tests']:
 try:
  with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):v=fn(t['x'])
  passed+=int(type(v) is int and v==t['expected'])
 except BaseException as ex:errors[type(ex).__name__]=errors.get(type(ex).__name__,0)+1
 done+=1
 emit({'status':'progress','completed_cases':done,'passed':passed,'cpu_seconds':time.process_time()-cpu0,'wall_seconds':time.monotonic()-wall0})
emit({'status':'complete','completed_cases':done,'passed':passed,'errors':errors,'cpu_seconds':time.process_time()-cpu0,'wall_seconds':time.monotonic()-wall0})
'''

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]

def run(code,tests,mode):
    name='coderl-time-probe-'+uuid.uuid4().hex[:16]
    cmd=['docker','run','--rm','--name',name,'-i','--network','none','--memory','256m','--cpus','1','--pids-limit','32',
         '--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--user','65534:65534',
         '--tmpfs','/tmp:rw,nosuid,size=16m','--workdir','/tmp','-e','PYTHONDONTWRITEBYTECODE=1',
         '--entrypoint','python',IMAGE,'-I','-c',DRIVER]
    started=time.monotonic();raw='';exitcode=None;timedout=False
    try:
        try:
            r=subprocess.run(cmd,input=json.dumps({'code':code,'tests':tests,'cpu_limit':mode=='cpu8_wall30'}),text=True,capture_output=True,timeout=8 if mode=='wall8' else 30)
            raw=r.stdout;exitcode=r.returncode
        except subprocess.TimeoutExpired as ex:
            timedout=True;raw=ex.stdout or ''
            if isinstance(raw,bytes):raw=raw.decode(errors='replace')
        messages=[json.loads(x[len('__EXP026_TIME__'):]) for x in raw.splitlines() if x.startswith('__EXP026_TIME__')]
        if not timedout and exitcode not in (0,124,137,152):raise RuntimeError('Docker infrastructure exit '+str(exitcode))
        return {'mode':mode,'wall_guard_expired':timedout,'host_wall_seconds':round(time.monotonic()-started,4),
                'container_exit':exitcode,'last_evidence':messages[-1] if messages else None,
                'status':'wall_guard_indeterminate' if timedout else (messages[-1]['status'] if messages else 'unclassified_resource_exit')}
    finally:
        subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=15)


def main():
    p=argparse.ArgumentParser();p.add_argument('--train-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    source=a.train_root/'artifacts/train017/frozen24/train.jsonl'
    if sha(source)!=TRAIN_SHA:raise ValueError('training input drift')
    tasks={x['task_id']:x for x in read(source)}
    root=a.train_root/'artifacts/exp026/depfix_seed43'
    candidates={r['attempt_id']:r for r in read(root/'local_candidates.jsonl')}
    rewards=read(root/'reward_attempts.jsonl')
    report={'experiment':'EXP026-timeout-replay','not_a_training_update':True,'source_sha256':sha(__file__),
            'train_sha256':TRAIN_SHA,'records':[],'warning':'Wall8 includes Docker start/IPC and scheduling. CPU8 differs in budget semantics. No cause of timing variation is assumed.'}
    for repeat in range(2):
        for i in (22,23):
            c=candidates[i];r=rewards[i]
            if c['code_sha256']!=r['code_sha256'] or hashlib.sha256(c['code'].encode()).hexdigest()!=c['code_sha256']:raise ValueError('candidate mutation')
            tests=validate_tests(tasks[c['task_id']]['test_cases'])
            for mode in ('wall8','cpu8_wall30'):
                item={'attempt_id':i,'repeat':repeat,'task_id':c['task_id'],'code_sha256':c['code_sha256'],'num_tests':len(tests),**run(c['code'],tests,mode)}
                report['records'].append(item);print(json.dumps(item),flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
