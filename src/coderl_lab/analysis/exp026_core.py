"""EXP-026 pure diagnostics. No candidate execution or hidden-test access.

Dependency is an OBSERVED no-dependency-error fraction, not a completeness proof.
The advantage calculation follows installed TRL's sample std and epsilon=1e-4;
float64 offline arithmetic is a diagnostic, not a bitwise gradient replay.
"""
from __future__ import annotations
import math
from statistics import mean, stdev

DEPENDENCY_ERRORS=frozenset({'NameError','UnboundLocalError','ImportError','ModuleNotFoundError'})

def features(probe:dict)->dict:
    if type(probe.get('loaded')) is not bool or not isinstance(probe.get('cases'),list):
        raise ValueError('malformed probe header')
    cases=probe['cases']
    if not probe['loaded']:
        if cases or not isinstance(probe.get('load_error'),str):
            raise ValueError('failed load must contain error and no cases')
        return {'reward':0.,'executable_fraction':0.,'dependency_fraction':0.,
                'semantic_fraction':0.,'verdict':probe['load_error']}
    if not cases:raise ValueError('successful load requires nonempty per-case evidence')
    errors=[]
    for r in cases:
        if (type(r.get('executed')) is not bool or type(r.get('correct')) is not bool
            or not isinstance(r.get('error'),str)):
            raise ValueError('malformed per-case verdict')
        if r['correct'] and not r['executed']:raise ValueError('correct but not executed')
        if r['executed'] != (r['error']=='none'):raise ValueError('error/execution contradiction')
        dep=r.get('dependency_error',r['error'] in DEPENDENCY_ERRORS)
        if type(dep) is not bool or (dep and r['executed']):raise ValueError('bad dependency flag')
        errors.append(dep)
    e=mean(r['executed'] for r in cases)
    d=1-mean(errors)
    s=mean(r['correct'] for r in cases)
    return {'reward':round(.1*e+.1*d+.8*s,6),'executable_fraction':e,
            'dependency_fraction':d,'semantic_fraction':s,'verdict':'completed'}

def advantage(values:list[float],eps:float=1e-4)->list[float]:
    if len(values)<2 or eps<0 or any(not math.isfinite(x) for x in values):
        raise ValueError('invalid advantage inputs')
    mu=mean(values);sd=stdev(values)
    if sd==0:return [0.]*len(values)
    return [(r-mu)/(sd+eps) for r in values]

def groups(rows:list[dict],size:int=4):
    if not rows or len(rows)%size:raise ValueError('incomplete reward groups')
    for i in range(0,len(rows),size):
        g=rows[i:i+size]
        if len({r['task_id'] for r in g})!=1:raise ValueError('mixed task group')
        yield g

def ablation(rows:list[dict])->dict:
    gs=list(groups(rows));out={'groups':len(gs),'samples':len(rows)}
    for name,key in [('execution','executable_fraction'),('dependency','dependency_fraction'),('semantic','semantic_fraction')]:
        out['varying_'+name+'_groups']=sum(len({r[key] for r in g})>1 for g in gs)
    for name,weights in [('remove_dependency',(.1,0.,.8)),('remove_execution',(0.,.1,.8)),('semantic_only',(0.,0.,.8))]:
        changes=[];flips=0;rank_changes=0;affected=0
        for g in gs:
            baseline=[r['reward'] for r in g]
            counter=[round(sum(w*r[k] for w,k in zip(weights,('executable_fraction','dependency_fraction','semantic_fraction'))),6) for r in g]
            a=advantage(baseline);b=advantage(counter)
            delta=max(abs(x-y) for x,y in zip(a,b));changes.append(delta)
            flips+=sum(x*y<0 and abs(x)>1e-8 and abs(y)>1e-8 for x,y in zip(a,b))
            rank_changes+=sum((baseline[i]-baseline[j])*(counter[i]-counter[j])< -1e-12 for i in range(4) for j in range(i+1,4))
            affected+=int(delta>1e-5)
        out[name]={'max_advantage_delta':max(changes),'mean_group_max_delta':mean(changes),
                   'groups_delta_above_1e5':affected,'sign_flips':flips,'pairwise_rank_reversals':rank_changes}
    return out
