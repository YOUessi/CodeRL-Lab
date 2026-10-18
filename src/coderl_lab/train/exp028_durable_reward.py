"""Durable evidence wrapper; delegates all rewards to the unchanged matched code.

Each four-candidate group is saved on Tang before any execution. Every probe has
start/completion/failure records. This process-local wrapper is for the serial
single-GPU trainer; it does not offer a partial group to the optimizer.
"""
from __future__ import annotations
import hashlib,os
from pathlib import Path
from coderl_lab.train import exp027_matched_reward as matched

_BASE_FACTORY=matched.reward_factory

def write_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:
        f.write(matched.canonical(value)+'\n');f.flush();os.fsync(f.fileno())


def append(path,value):
    with path.open('a',encoding='utf8') as f:
        f.write(matched.canonical(value)+'\n');f.flush();os.fsync(f.fileno())


def reward_factory(logfile:Path,variant:str,cache_root:Path):
    logfile=Path(logfile)
    if (logfile.parent/'pending_groups').exists():raise FileExistsError('partial group evidence: explicit recovery required')
    original_callback=_BASE_FACTORY(logfile,variant,cache_root)
    group=0
    def callback(prompts,completions,completion_ids=None,task_id=None,test_cases=None,**kwargs):
        nonlocal group
        if (task_id is None or test_cases is None or not(len(prompts)==len(completions)==len(task_id)==len(test_cases)==4)
            or len(set(task_id))!=1):raise ValueError('require complete four-candidate group')
        prepared={'status':'prepared_before_execution','group_id':group,'variant':variant,
                  'task_ids':task_id,'completions':completions,'test_cases':test_cases,
                  'prompts':prompts,'optimizer_step_committed':False}
        write_new(logfile.parent/'pending_groups'/f'group_{group:04d}.json',prepared)
        previous_observe=matched.observe
        def observed(cache,code,tests):
            meta={'group_id':group,'task_id':task_id[0],
                  'code_sha256':hashlib.sha256(code.encode()).hexdigest(),'tests_sha256':matched.digest_value(tests)}
            events=logfile.parent/'probe_events.jsonl'
            append(events,{**meta,'status':'started'})
            try:r=previous_observe(cache,code,tests)
            except Exception as e:
                append(events,{**meta,'status':'failed','error_type':type(e).__name__,'reason':str(e)})
                raise
            append(events,{**meta,'status':'completed','probe_key':r['key'],'cache_hit':r['cache_hit'],
                           'probe_sha256':r['probe_sha256'],'resource_outcome':r['probe'].get('load_error')})
            return r
        matched.observe=observed
        try:
            result=original_callback(prompts,completions,completion_ids=completion_ids,task_id=task_id,test_cases=test_cases,**kwargs)
        finally:matched.observe=previous_observe
        write_new(logfile.parent/'reward_groups_ready'/f'group_{group:04d}.json',
                  {'group_id':group,'status':'reward_group_ready','rewards':result,
                   'optimizer_step_not_asserted_by_callback':True})
        group+=1
        return result
    callback.__name__=original_callback.__name__
    return callback
