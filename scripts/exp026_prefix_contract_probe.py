"""Additional post-hoc contract probe; no hidden tests are read or changed.

The short public description does not explicitly define N. Prefix-of-length-N
is tested as a NAMED convention rather than an unbiased semantic ground truth.
"""
import argparse,json
from pathlib import Path
from coderl_lab.analysis.exp026_runner import load_bundle,probe,rows,sha,write_once,PUBLIC_SHA
from coderl_lab.analysis.exp021_seed42_docker_score import extract_code

SUITE=r'''
import itertools
for n in range(5):
 for a in itertools.product((-2,-1,0,1,2),repeat=n):
  for N in range(n+1):
   expected=sum((a[i]^a[j])%2==1 for i in range(N) for j in range(i+1,N))
   check((list(a),N),expected,kind='explicit_prefix_N_convention')
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--eval-root',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    pp=a.eval_root/'artifacts/exp024/frozen_new96/public96_prompts.jsonl'
    if sha(pp)!=PUBLIC_SHA:raise ValueError('cohort drift')
    pub={r['task_id']:r for r in rows(pp)};result={'experiment':'EXP026-prefix-contract','scope':'posthoc explicit convention, not new benchmark','cases':[],'source_sha256':sha(__file__)}
    for s in (43,44):
        for arm,path in [('old',a.eval_root/f'artifacts/exp025/old_seed{s}_new96'),('strict',a.eval_root/f'artifacts/exp024/eval_seed{s}_new96')]:
            g,v,identity=load_bundle(path,pub)
            for i in (0,1):
                code,_=extract_code(g['MBPPPlus/556',i,'grpo']['raw_code'])
                result['cases'].append({'seed':s,'arm':arm,'sample':i,'result':probe(code,pub['MBPPPlus/556']['entry_point'],SUITE)})
    write_once(a.out,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
