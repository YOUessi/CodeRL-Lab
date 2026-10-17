"""End-to-end diagnostics over frozen artifacts. Candidate execution: Docker ONLY.

Post-hoc independent probes are NOT new benchmark scores. Original benchmarks,
model outputs and hidden test files are never modified. No hidden tests are read.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, subprocess, sys, time, uuid
from pathlib import Path
from coderl_lab.analysis.exp026_core import ablation

PUBLIC_SHA='6438a8abdfc5ec19edd98f717e03d17b2c8a13f5fbe25f07a63db7ee030755f7'
IMAGE='sha256:8734cd8b7868a1b0205384477cfdef64753ff979eb45d0249f60dc31a64c74ea'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def write_once(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')

def load_bundle(p,ids):
    gen=json.loads((p/'generation_summary.json').read_text());sc=json.loads((p/'score_summary.json').read_text())
    if (gen.get('status')!='all_samples_generated_frozen' or sc.get('status')!='completed_pinned_MBPPPlus_96_tasks'
        or gen['checkpoint_sha256']!=sha(p/'checkpoint.jsonl')
        or sc['candidate_scoring_checkpoint_sha256']!=sha(p/'docker_score_checkpoint.jsonl')
        or gen['public_sha256']!=PUBLIC_SHA):raise ValueError('unfrozen or changed inputs')
    g={};v={}
    for r in rows(p/'checkpoint.jsonl'):
        k=(r['task_id'],r['sample_id'],r['arm'])
        if k in g or hashlib.sha256(r['raw_code'].encode()).hexdigest()!=r['code_sha256']:raise ValueError('generation mutation')
        g[k]=r
    for r in rows(p/'docker_score_checkpoint.jsonl'):
        k=(r['task_id'],r['sample_id'],r['arm'])
        if k in v or k not in g or r['code_sha256']!=g[k]['code_sha256']:raise ValueError('score mismatch')
        v[k]=r
    expected={(q,i,a) for q in ids for i in range(4) for a in ('sft','grpo')}
    if set(g)!=expected or set(v)!=expected:raise ValueError('incomplete pairs')
    return g,v,{'generation_sha256':sha(p/'checkpoint.jsonl'),'scoring_sha256':sha(p/'docker_score_checkpoint.jsonl'),'models':gen['model_adapters_sha256']}

# All extra cases below are explicitly specified diagnostic domains, not hidden tests.
SUITES={
 'MBPPPlus/732':r'''
import itertools
for n in range(6):
 for t in itertools.product('a ,.',repeat=n):
  x=''.join(t);check((x,),''.join(':' if c in ' ,.' else c for c in x))
''',
 'MBPPPlus/556':r'''
import itertools
for n in range(5):
 for a in itertools.product((-2,-1,0,1,2),repeat=n):
  check((list(a),len(a)),sum((a[i]^a[j])%2==1 for i in range(n) for j in range(i+1,n)))
''',
 'MBPPPlus/137':r'''
import itertools
for n in range(1,5):
 for a in itertools.product((-1,0,1),repeat=n):
  z=a.count(0)
  if z<n:check((list(a),),z/(n-z))
# The public task does not specify zero-denominator behavior: report separately.
for a in ([],[0],[0,0]):check((a,),float('inf'),kind='explicit_infinity_convention')
''',
 'MBPPPlus/409':r'''
for a in [[(2,3)],[(2,3),(-2,4)],[(0,9),(3,3)],[(-5,-2),(1,-3),(0,2)]]:
 check((a,),min(x*y for x,y in a))
''',
 'MBPPPlus/432':r'''
for a in (1,2,5):
 for b in (1,4,7):
  for h in (1,2,5):check((a,b,h),(a+b)/2)
# Nonpositive height is not a nondegenerate geometric trapezium.
for h in (0,-1):check((2,4,h),3.,kind='out_of_geometric_domain')
''',
 'MBPPPlus/757':r'''
import itertools
for n in range(5):
 for a in itertools.product(('ab','ba','aa','bb'),repeat=n):
  check((list(a),),sum(a[i]==a[j][::-1] for i in range(n) for j in range(i+1,n)))
''',
 'MBPPPlus/12':r'''
for m in ([],[[1]],[[2,0],[1,0]],[[0,2],[1,1],[-1,0]],[[],[1],[0,-2]]):
 check((m,),sorted(m,key=lambda row:sum(row)))
''',
 'MBPPPlus/124':r'''
# Public signature specifies two parameters but the short description does not
# state their combination; test arity only, not an invented semantic oracle.
import inspect
try:
 inspect.signature(fn).bind(1,2);results.append({'kind':'signature_only','passed':True,'error':'none'})
except Exception as ex:results.append({'kind':'signature_only','passed':False,'error':type(ex).__name__})
''',
 'MBPPPlus/297':r'''
for a,b in [([],[]),([1],[1]),([[1,2],[3]],[1,2,3]),([1,[2,[3]],[]],[1,2,3]),([[],[[[]]]],[])]:
 check((a,),b)
''',
 'MBPPPlus/95':r'''
for a in ([[1]],[[1,2],[]],[[1],[2,3],[4,5,6]],[[None],[1,2]]):
 check((a,),min(map(len,a)))
''',
}
DRIVER=r'''
import json,sys,contextlib,io,math,collections
p=json.load(sys.stdin);space={'__name__':'__candidate__'};results=[]
try:
 with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
  exec(compile(p['code'],'<candidate>','exec'),space,space)
 fn=space[p['entry']]
 if not callable(fn):raise TypeError('entry not callable')
except BaseException as ex:
 out={'loaded':False,'load_error':type(ex).__name__,'domains':{}}
else:
 def check(args,expected,kind='defined_domain'):
  try:
   with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):actual=fn(*args)
   ok=actual==expected
   results.append({'kind':kind,'passed':bool(ok),'error':'none' if ok else 'mismatch'})
  except BaseException as ex:results.append({'kind':kind,'passed':False,'error':type(ex).__name__})
 ctx={'fn':fn,'check':check,'results':results}
 exec(compile(p['suite'],'<diagnostic_suite>','exec'),ctx,ctx)
 domains={}
 for kind in sorted({r['kind'] for r in results}):
  rs=[r for r in results if r['kind']==kind]
  domains[kind]={'cases':len(rs),'passed':sum(r['passed'] for r in rs),'errors':dict(collections.Counter(r['error'] for r in rs if not r['passed']))}
 out={'loaded':True,'load_error':None,'domains':domains}
print('__EXP026__'+json.dumps(out))
'''

def probe(code,entry,suite):
    name='coderl-exp026-'+uuid.uuid4().hex[:16]
    cmd=['docker','run','--rm','--name',name,'-i','--network','none','--read-only','--cap-drop','ALL',
         '--security-opt','no-new-privileges','--user','65534:65534','--memory','512m','--cpus','1','--pids-limit','32',
         '--tmpfs','/tmp:rw,nosuid,size=32m','-e','PYTHONDONTWRITEBYTECODE=1','--entrypoint','/usr/bin/python3',IMAGE,'-I','-c',DRIVER]
    try:
        try:r=subprocess.run(cmd,input=json.dumps({'code':code,'entry':entry,'suite':suite}),capture_output=True,text=True,timeout=20)
        except subprocess.TimeoutExpired:return {'loaded':False,'load_error':'diagnostic_timeout','domains':{}}
        if r.returncode!=0:raise RuntimeError(f'Docker diagnostic infrastructure exit {r.returncode}: {r.stderr[-200:]}')
        lines=[x[len('__EXP026__'):] for x in r.stdout.splitlines() if x.startswith('__EXP026__')]
        if len(lines)!=1:raise RuntimeError('invalid diagnostic verdict envelope')
        return json.loads(lines[0])
    finally:
        # Ensure Docker children never survive a host-side timeout.
        subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=10)

def initial(train,ev,out):
    from coderl_lab.analysis.exp021_seed42_docker_score import extract_code
    pubpath=ev/'artifacts/exp024/frozen_new96/public96_prompts.jsonl'
    if sha(pubpath)!=PUBLIC_SHA:raise ValueError('public cohort changed')
    pub={r['task_id']:r for r in rows(pubpath)}
    report={'experiment':'EXP026','private_tests_read':False,'protocol':'posthoc complete flip audit and fixed-rollout reward diagnostics',
            'cases':[],'signals':{},'sources':{},'code_sha256':sha(__file__),'limitations':'Not an unbiased new benchmark; convention-only cases and interface checks are reported separately.'}
    for s in (43,44):
        pairs={}
        for arm,p in [('old',ev/f'artifacts/exp025/old_seed{s}_new96'),('strict',ev/f'artifacts/exp024/eval_seed{s}_new96')]:
            g,v,identity=load_bundle(p,pub)
            pairs[arm]=(g,v);report['sources'][f'{arm}_{s}']=identity
        for q in pub:
            for i in range(4):
                if pairs['old'][0][q,i,'sft']['code_sha256']!=pairs['strict'][0][q,i,'sft']['code_sha256']:
                    raise ValueError('SFT reference decode drift')
                if pairs['old'][1][q,i,'sft']['passed']!=pairs['strict'][1][q,i,'sft']['passed']:
                    raise ValueError('SFT reference verdict drift')
                a,b=(pairs[arm][1][q,i,'grpo'] for arm in ('old','strict'))
                if a['passed']==b['passed']:continue
                if q not in SUITES:raise ValueError('unreviewed flip task '+q)
                item={'seed':s,'task_id':q,'sample':i,'direction':'repair' if b['passed'] else 'regression',
                      'suite_sha256':hashlib.sha256(SUITES[q].encode()).hexdigest(),'outcomes':{}}
                for arm in ('old','strict'):
                    code,norm=extract_code(pairs[arm][0][q,i,'grpo']['raw_code'])
                    result=probe(code,pub[q]['entry_point'],SUITES[q])
                    item['outcomes'][arm]={'code_sha256':hashlib.sha256(code.encode()).hexdigest(),'normalizer':norm,**result}
                    if arm=='strict' and q=='MBPPPlus/732':
                        corrected=code.replace("r'[ ,.]+'","r'[ ,.]'")
                        if corrected==code:raise ValueError('regex intervention not applicable')
                        item['regex_control']=probe(corrected,pub[q]['entry_point'],SUITES[q])
                    if arm=='strict' and q=='MBPPPlus/757':
                        item['typing_control']=probe('from typing import List\n'+code,pub[q]['entry_point'],SUITES[q])
                report['cases'].append(item)
                print('DIAGNOSED',s,q,i,flush=True)
        p=train/f'artifacts/exp024/strict_seed{s}'
        log=p/'reward_attempts.jsonl';summary=json.loads((p/'summary.json').read_text())
        if summary['reward_attempt_log_sha256']!=sha(log):raise ValueError('training reward log changed')
        r=rows(log)
        if len(r)!=128:raise ValueError('training log incomplete')
        report['signals'][str(s)]={**ablation(r),'reward_log_sha256':sha(log)}
    if len(report['cases'])!=16:raise ValueError('expected all 16 flips, not cherry-picked subset')
    from coderl_lab.train import exp024_reward,exp026_reward
    tests=[{'x':str(i),'expected':i} for i in range(10)]
    samples={
        'missing_module':'def solve(x):\n import exp026_nonexistent_module\n return int(x)',
        'unbound_local':'def solve(x):\n a=a+1\n return int(x)',
        'loadtime_name':'def solve(x: List[int]): return 0',
        'valid_correct':'def solve(x): return int(x)',
        'valid_wrong':'def solve(x): return 0',
    }
    report['real_docker_contracts']={label:{'old':exp024_reward.score(code,tests),'fixed':exp026_reward.score(code,tests)} for label,code in samples.items()}
    for label in ('missing_module','unbound_local'):
        z=report['real_docker_contracts'][label]
        if z['old']['reward']!=.1 or z['fixed']['reward']!=0.:raise AssertionError('dependency regression test failed')
    if report['real_docker_contracts']['valid_correct']['fixed']['reward']!=1.:raise AssertionError('lost semantic reward')
    report['status']='diagnostics_and_docker_contracts_complete'
    write_once(out/'diagnostics.json',report)
    print('DIAGNOSTICS_COMPLETE',len(report['cases']),flush=True)

def post_train(train,out):
    import torch
    from safetensors.torch import load_file
    report={'experiment':'EXP026','status':'training_audit_complete','seeds':{},'external_evaluation':'not_repeated_in_this_stage'}
    for s in (43,44):
        p=out/f'depfix_seed{s}';d=json.loads((p/'summary.json').read_text());log=rows(p/'reward_attempts.jsonl')
        old=train/f'artifacts/exp024/strict_seed{s}'
        if (d['status']!='real_GPU_GRPO_completed' or d['global_step']!=32 or len(log)!=128
            or d['saved_adapter_sha256']!=sha(p/'adapter_model.safetensors')
            or d['reward_attempt_log_sha256']!=sha(p/'reward_attempts.jsonl')):raise ValueError('training not verified')
        a=load_file(str(old/'adapter_model.safetensors'));b=load_file(str(p/'adapter_model.safetensors'))
        if set(a)!=set(b):raise ValueError('adapter keys changed')
        exact=all(torch.equal(a[k],b[k]) for k in a)
        changed=sum(int((a[k]!=b[k]).sum()) for k in a)
        report['seeds'][str(s)]={'adapter_sha256':d['saved_adapter_sha256'],
            'matches_previous_strict_parameters':exact,'changed_elements_vs_previous_strict':changed,
            'changed_elements_vs_SFT':d['initial_SFT_parameter_elements_changed_after_grpo'],
            'same_rollout_reward_changed_samples':sum(r.get('legacy_reward_same_probe',r['reward'])!=r['reward'] for r in log),
            'signals':ablation(log),'active_reward_source_sha256':d['active_reward_source_sha256'],
            'local_candidate_sha256':sha(p/'local_candidates.jsonl')}
    write_once(out/'training_audit.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--train-root',type=Path,required=True);p.add_argument('--eval-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--post-train',action='store_true');a=p.parse_args()
    if a.post_train:post_train(a.train_root,a.out)
    elif (a.out/'diagnostics.json').exists():raise FileExistsError('existing diagnostics; do not silently replace')
    else:initial(a.train_root,a.eval_root,a.out)

if __name__=='__main__':main()
