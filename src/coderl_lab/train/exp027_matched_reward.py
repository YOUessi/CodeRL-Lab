"""Matched EXP027 observations. Only completed, validated Docker probes are cached.

The shared execution ledger defines a fixed observational environment, not a
claim of universal runtime determinism. Candidate source stays on Tang.
"""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path
from coderl_lab.train import exp027_guarded_probe as guarded
from coderl_lab.train import exp024_reward as original
from coderl_lab.train import train014b_reward_sandbox as sandbox
from coderl_lab.analysis.exp026_core import features

VARIANTS=('legacy','corrected')
WALL_SECONDS=60.

def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)

def digest_value(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def protocol():
    return {'id':'EXP027-matched-completed-probe-v1','wall_seconds':WALL_SECONDS,
            'image_id':sandbox.IMAGE_ID,'driver_sha256':hashlib.sha256(original.DRIVER.encode()).hexdigest(),
            'guard_sha256':file_sha(guarded.__file__),'validation_sha256':file_sha(sandbox.__file__),
            'features_source_sha256':file_sha(__import__('coderl_lab.analysis.exp026_core',fromlist=['features']).__file__),
            'unscorable':'abort_no_zero_no_cache','reuse':'same_code_tests_protocol_completed_result'}

def rewards(result):
    f=features(result)
    if result['loaded']:
        old_d=sum(x['error']!='NameError' for x in result['cases'])/len(result['cases'])
        legacy=round(.1*f['executable_fraction']+.1*old_d+.8*f['semantic_fraction'],6)
    else:old_d=0.;legacy=0.
    return {**f,'legacy_reward':legacy,'corrected_reward':f['reward'],
            'legacy_dependency_fraction':old_d}

def validate_record(rec,key,identity,tests):
    if rec.get('key')!=key or rec.get('protocol')!=identity or rec.get('status')!='complete':
        raise ValueError('cache identity/status mismatch')
    result=rec['probe'];features(result)
    if result['loaded'] and len(result['cases'])!=len(tests):raise ValueError('cache case count')
    if rec.get('probe_sha256')!=digest_value(result):raise ValueError('cache content hash mismatch')
    return result

def observe(cache_root:Path,code:str,tests):
    tests=sandbox.validate_tests(tests);identity=protocol()
    key=digest_value({'code':code,'tests':tests,'protocol':identity})
    cache_root=Path(cache_root);cache_root.mkdir(parents=True,exist_ok=True)
    path=cache_root/(key+'.json')
    with (cache_root/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():
            record=json.loads(path.read_text());result=validate_record(record,key,identity,tests)
            return {'key':key,'probe':result,'cache_hit':True,'probe_sha256':record['probe_sha256']}
        started=time.monotonic()
        result=guarded.docker_probe(code,tests,wall_seconds=WALL_SECONDS)
        features(result)
        if result['loaded'] and len(result['cases'])!=len(tests):raise ValueError('incomplete observation')
        record={'key':key,'protocol':identity,'status':'complete','probe':result,
                'probe_sha256':digest_value(result),'elapsed_seconds':time.monotonic()-started}
        # Write only after execution and cleanup have both been confirmed.
        tmp=path.with_suffix('.tmp')
        with tmp.open('w') as f:
            f.write(canonical(record)+'\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
        return {'key':key,'probe':result,'cache_hit':False,'probe_sha256':record['probe_sha256']}

def reward_factory(logfile:Path,variant:str,cache_root:Path):
    if variant not in VARIANTS:raise ValueError('unknown reward variant')
    logfile=Path(logfile);logfile.parent.mkdir(parents=True,exist_ok=True)
    candidates=logfile.with_name('local_candidates.jsonl')
    if logfile.exists() or candidates.exists():raise FileExistsError('refuse mixed reward logs')
    attempt=0
    def callback(prompts,completions,completion_ids=None,task_id=None,test_cases=None,**kwargs):
        nonlocal attempt
        if (task_id is None or test_cases is None or
            not(len(prompts)==len(completions)==len(task_id)==len(test_cases)) or len(completions)!=4 or len(set(task_id))!=1):
            raise ValueError('require one complete four-candidate task group')
        pending=[];sources=[]
        try:
            for i,(q,c,t) in enumerate(zip(task_id,completions,test_cases)):
                tests=sandbox.validate_tests(t);code,norm=sandbox.normalize_code(c)
                valid,entry=sandbox.check_code(code)
                meta={'attempt_id':attempt+i,'group_id':attempt//4,'task_id':q,'variant':variant,
                      'normalizer':norm,'code_sha256':hashlib.sha256(code.encode()).hexdigest(),
                      'tests_sha256':digest_value(tests),'syntax_and_entry':bool(valid and entry)}
                if valid and entry:
                    obs=observe(cache_root,code,tests);f=rewards(obs['probe'])
                    row={**meta,**f,'reward':f[variant+'_reward'],'cache_hit':obs['cache_hit'],
                         'probe_key':obs['key'],'probe_sha256':obs['probe_sha256']}
                else:
                    row={**meta,'reward':0.,'legacy_reward':0.,'corrected_reward':0.,
                         'executable_fraction':0.,'dependency_fraction':0.,'semantic_fraction':0.,
                         'legacy_dependency_fraction':0.,'verdict':'invalid_code','cache_hit':False,
                         'probe_key':None,'probe_sha256':None}
                pending.append(row);sources.append({**meta,'code':code})
        except Exception as exc:
            marker=logfile.with_name('unscorable_abort.json')
            marker.write_text(canonical({'group_id':attempt//4,'task_ids':task_id,
                                        'error_type':type(exc).__name__,'reason':str(exc),
                                        'reward_group_committed':False})+'\n')
            raise
        # No group is offered to GRPO until every observation is trustworthy.
        with candidates.open('a',encoding='utf8') as f:
            f.writelines(canonical(x)+'\n' for x in sources);f.flush();os.fsync(f.fileno())
        with logfile.open('a',encoding='utf8') as f:
            f.writelines(canonical(x)+'\n' for x in pending);f.flush();os.fsync(f.fileno())
        attempt+=len(pending)
        return [x['reward'] for x in pending]
    callback.__name__='exp027_'+variant+'_matched_reward'
    return callback
