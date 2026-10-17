"""Uniform EXP027 evaluation, no reuse of historical verdicts.

CPU budget exhaustion is a separate resource-failure outcome under a specified
budget; host timeout/unknown exit is indeterminate and aborts, never score zero.
Historical answer generation may be reused only for exact adapter and recipe.
"""
from __future__ import annotations
import collections
import hashlib
import json
import subprocess
import uuid
from pathlib import Path
import numpy as np
from coderl_lab.analysis.exp021_seed42_docker_score import DRIVER as BASE_DRIVER,IMAGE_ID,extract_code,ast_valid
from coderl_lab.train.exp027_guarded_probe import SandboxIndeterminate

PUBLIC_SHA='6438a8abdfc5ec19edd98f717e03d17b2c8a13f5fbe25f07a63db7ee030755f7'
PRIVATE_SHA='e945ac3b28fbecc306e24468b796ec883758d10588c5ab73fd37362518259060'
REVISION='a09a35458c702b33eeacc393d103063234e8bc28'
SFT_SHA='25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8'
GENERATION={'max_new_tokens':256,'temperature':.8,'top_p':.95,'do_sample':True,'base_quantization':'NF4_double_quant_BF16','seed':42}
CPU_SECONDS=15
WALL_SECONDS=60
CPU_PREFIX='''import resource,signal,os

def _exp027_cpu_exhausted(signum,frame):
    os.write(1,b"__EXP027_CPU_BUDGET__\\n")
    os._exit(124)
signal.signal(signal.SIGXCPU,_exp027_cpu_exhausted)
resource.setrlimit(resource.RLIMIT_CPU,({cpu},{hard}))
'''
DRIVER=CPU_PREFIX.format(cpu=CPU_SECONDS,hard=CPU_SECONDS+1)+BASE_DRIVER

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def objsha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]

def protocol():
    return {'experiment':'EXP027-matched-eval','image_id':IMAGE_ID,'driver_sha256':hashlib.sha256(DRIVER.encode()).hexdigest(),
            'cpu_seconds_per_process':CPU_SECONDS,'host_wall_guard_seconds':WALL_SECONDS,
            'wall_or_unknown_exit':'indeterminate_abort','cpu_budget_exit':'resource_failure_distinct_from_semantic_error'}

def validate_recipe(s):
    if s['public_sha256']!=PUBLIC_SHA or s['model_revision']!=REVISION:raise ValueError('source cohort/model mismatch')
    if any(s['configuration'].get(k)!=v for k,v in GENERATION.items()):raise ValueError('source sampler recipe mismatch')
    if any(s['model_inference_adapter_dtype'].get(k)!=['torch.float32'] for k in ('sft','grpo')):raise ValueError('source inference dtype mismatch')

def decode_exit(code,stdout):
    cpu=stdout.splitlines().count('__EXP027_CPU_BUDGET__')
    if code==124 and cpu==1:return {'passed':False,'reason':'cpu_budget_exceeded','resource_limited':True}
    if code!=0 or cpu:raise SandboxIndeterminate('unclassified_container_exit_'+str(code))
    messages=[x[len('__EXP019B__'):] for x in stdout.splitlines() if x.startswith('__EXP019B__')]
    if len(messages)!=1:raise SandboxIndeterminate('nonunique_evaluation_verdict')
    try:r=json.loads(messages[0])
    except ValueError as exc:raise SandboxIndeterminate('invalid_evaluation_json') from exc
    if type(r.get('passed')) is not bool or not isinstance(r.get('reason'),str):raise SandboxIndeterminate('malformed_evaluation_verdict')
    if r['passed']!=(r['reason']=='none'):raise SandboxIndeterminate('contradictory_evaluation_verdict')
    return {**r,'resource_limited':False}

def execute(code,test,entry,cpu_seconds=CPU_SECONDS,wall_seconds=WALL_SECONDS):
    if not 1<=cpu_seconds<=15 or not 0<wall_seconds<=60:raise ValueError('unsafe evaluation bounds')
    driver=CPU_PREFIX.format(cpu=cpu_seconds,hard=cpu_seconds+1)+BASE_DRIVER
    name='coderl-eval027-'+uuid.uuid4().hex[:16]
    cmd=['docker','run','--rm','--name',name,'-i','--network','none','--memory','1g','--cpus','1',
         '--pids-limit','64','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges',
         '--user','65534:65534','--tmpfs','/tmp:rw,nosuid,size=128m',
         '-e','PYTHONDONTWRITEBYTECODE=1','-e','OPENBLAS_NUM_THREADS=1',
         '--entrypoint','/usr/bin/python3',IMAGE_ID,'-c',driver]
    try:
        try:p=subprocess.run(cmd,input=json.dumps({'code':code,'test':test,'entry_point':entry}),text=True,capture_output=True,timeout=wall_seconds)
        except subprocess.TimeoutExpired as exc:raise SandboxIndeterminate('wall_expired_no_verdict') from exc
        return decode_exit(p.returncode,p.stdout)
    finally:
        try:c=subprocess.run(['docker','rm','-f',name],text=True,capture_output=True,timeout=15)
        except (subprocess.SubprocessError,OSError) as exc:raise SandboxIndeterminate('cleanup_unconfirmed') from exc
        if c.returncode and 'No such container' not in c.stderr:raise SandboxIndeterminate('cleanup_failed')

def score_cached(cache,raw,oracle):
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    code,normalizer=extract_code(raw);syntax,entry=ast_valid(code,oracle['entry_point'])
    key=objsha({'code':code,'test':oracle['test'],'entry':oracle['entry_point'],'protocol':protocol()})
    path=cache/(key+'.json')
    if path.exists():
        d=json.loads(path.read_text())
        if d['key']!=key or d['sha256']!=objsha(d['verdict']):raise ValueError('evaluation cache drift')
        return {**d['verdict'],'cache_hit':True,'normalizer':normalizer,'probe_key':key}
    r=execute(code,oracle['test'],oracle['entry_point']) if syntax and entry else {'passed':False,'reason':'invalid_syntax_or_entry','resource_limited':False}
    with path.open('x') as f:json.dump({'key':key,'verdict':r,'sha256':objsha(r)},f,sort_keys=True)
    return {**r,'cache_hit':False,'normalizer':normalizer,'probe_key':key}

def matrix(rr):
    groups={}
    for r in rr:
        key=r['task_id'];i=r['sample_id']
        if type(i) is not int or i not in range(4):raise ValueError('invalid draw')
        if i in groups.setdefault(key,{}):raise ValueError('duplicate draw')
        groups[key][i]=bool(r['passed'])
    if any(set(v)!=set(range(4)) for v in groups.values()):raise ValueError('incomplete task draws')
    return {k:[v[i] for i in range(4)] for k,v in groups.items()}

def metrics(rr):
    x=matrix(rr)
    return {'correct':sum(sum(v) for v in x.values()),'candidate_total':len(rr),'first':sum(v[0] for v in x.values()),
            'coverage':sum(any(v) for v in x.values()),'task_count':len(x),'reasons':dict(collections.Counter(r['reason'] for r in rr))}

def compare(a,b,reps=20000):
    if set(a)!=set(b) or not a:raise ValueError('paired task mismatch')
    ids=sorted(a);d=[];up=down=0
    for q in ids:
        av=a[q];bv=b[q]
        if len(av)!=4 or len(bv)!=4:raise ValueError('four draws required')
        d.append([(sum(bv)-sum(av))/4,int(bv[0])-int(av[0]),int(any(bv))-int(any(av))])
        up+=sum(not x and y for x,y in zip(av,bv));down+=sum(x and not y for x,y in zip(av,bv))
    x=np.array(d);boot=x[np.random.default_rng(20261018).integers(0,len(ids),(reps,len(ids)))].mean(axis=1)
    return {'candidate_delta':float(x[:,0].mean()),'first_delta':float(x[:,1].mean()),'coverage_delta':float(x[:,2].mean()),
            'ci95':np.quantile(boot,[.025,.975],axis=0).tolist(),'repairs':int(up),'regressions':int(down),
            'resample_unit':'task','task_count':len(ids),'replicates':reps,'seed':20261018}
