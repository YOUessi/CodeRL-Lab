"""Correct dependency-family classification; all other EXP-024 weights unchanged.

Generated programs are evaluated only by the existing pinned Docker probe.
Historical EXP-024 files remain untouched. Candidate bodies are retained locally
for same-rollout replay, NEVER in exported experiment summaries.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from coderl_lab.analysis.exp026_core import features
from coderl_lab.train.exp024_reward import docker_probe
from coderl_lab.train.train014b_reward_sandbox import validate_tests, normalize_code, check_code

def score(completion, cases):
    tests=validate_tests(cases);code,normalizer=normalize_code(completion)
    valid,entry=check_code(code)
    meta={'normalizer':normalizer,'code_sha256':hashlib.sha256(code.encode()).hexdigest(),
          'syntax_and_entry':bool(valid and entry),'reward_version':'EXP026-dependency-family-v2'}
    if not valid or not entry:
        return {**meta,'reward':0.,'executable_fraction':0.,'dependency_fraction':0.,
                'semantic_fraction':0.,'verdict':'invalid_code'}
    probe=docker_probe(code,tests)
    fixed=features(probe)
    if probe['loaded']:
        legacy_d=sum(r['error']!='NameError' for r in probe['cases'])/len(probe['cases'])
        legacy_r=round(.1*fixed['executable_fraction']+.1*legacy_d+.8*fixed['semantic_fraction'],6)
    else:legacy_d=0.;legacy_r=0.
    return {**meta,**fixed,'legacy_reward_same_probe':legacy_r,
            'legacy_dependency_fraction':legacy_d}


def reward_factory(logfile:Path):
    logfile.parent.mkdir(parents=True,exist_ok=True)
    if logfile.exists():raise FileExistsError('refuse mixed/reused reward log')
    candidates=logfile.with_name('local_candidates.jsonl')
    if candidates.exists():raise FileExistsError(candidates)
    attempt=0
    def callback(prompts,completions,completion_ids=None,task_id=None,test_cases=None,**kwargs):
        nonlocal attempt
        if task_id is None or test_cases is None or not(len(prompts)==len(completions)==len(task_id)==len(test_cases)):
            raise ValueError('reward metadata count mismatch')
        values=[]
        for q,c,t in zip(task_id,completions,test_cases):
            outcome=score(c,t);code,_=normalize_code(c)
            with candidates.open('a',encoding='utf8') as f:
                f.write(json.dumps({'attempt_id':attempt,'task_id':q,'code_sha256':outcome['code_sha256'],'code':code},ensure_ascii=False)+'\n')
            with logfile.open('a',encoding='utf8') as f:
                f.write(json.dumps({'attempt_id':attempt,'task_id':q,**outcome},ensure_ascii=False)+'\n')
            values.append(outcome['reward']);attempt+=1
        return values
    callback.__name__='exp026_dependency_family_v2'
    return callback
