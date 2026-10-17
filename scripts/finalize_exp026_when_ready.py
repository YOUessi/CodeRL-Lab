"""Deterministic final stage of EXP026. No raw programs/test bodies are exported.

Waits only for the two explicitly registered evaluation outputs, validates every
checkpoint, computes task-cluster bootstrap and optionally creates one GitHub
report through the user's existing gh authentication. Never overwrites a report.
"""
from __future__ import annotations
import argparse,base64,json,subprocess,time
from pathlib import Path
import numpy as np
from coderl_lab.analysis.exp026_runner import load_bundle,rows,write_once,PUBLIC_SHA,sha

REPO='YOUessi/CodeRL-Lab'
BRANCH='exp/7b-grpo-multiseed-generalization-20261015'
REPORT='docs/experiment-logs/2026-10-18-exp026-paired-evaluation-final.md'

def summary(g,v,ids):
    return {'correct':sum(v[q,i,'grpo']['passed'] for q in ids for i in range(4)),
            'first':sum(v[q,0,'grpo']['passed'] for q in ids),
            'coverage':sum(any(v[q,i,'grpo']['passed'] for i in range(4)) for q in ids),
            'nameerror':sum(v[q,i,'grpo']['reason']=='NameError' for q in ids for i in range(4))}

def compare(a,b,ids):
    deltas=[];up=down=0
    for q in ids:
        av=[int(a[q,i,'grpo']['passed']) for i in range(4)]
        bv=[int(b[q,i,'grpo']['passed']) for i in range(4)]
        deltas.append([(sum(bv)-sum(av))/4,bv[0]-av[0],int(any(bv))-int(any(av))])
        up+=sum(x==0 and y==1 for x,y in zip(av,bv));down+=sum(x==1 and y==0 for x,y in zip(av,bv))
    x=np.array(deltas,dtype=float)
    indices=np.random.default_rng(20261018).integers(0,len(ids),size=(20000,len(ids)))
    boot=x[indices].mean(axis=1)
    ci=np.quantile(boot,[.025,.975],axis=0)
    return {'delta':x.mean(axis=0).tolist(),'ci95':ci.tolist(),'repair_pairs':up,'regression_pairs':down,
            'resample_unit':'task','bootstrap_seed':20261018,'replicates':20000,'warning':'nominal intervals; only two training seeds'}

def analyze(train,ev):
    public=ev/'artifacts/exp024/frozen_new96/public96_prompts.jsonl'
    if sha(public)!=PUBLIC_SHA:raise ValueError('cohort changed')
    ids=sorted(r['task_id'] for r in rows(public));result={'experiment':'EXP026-final-evaluation','seeds':{},'sources':{},'public_sha256':PUBLIC_SHA}
    for s in (43,44):
        bs={}
        for arm,p in [('old_full',ev/f'artifacts/exp025/old_seed{s}_new96'),('old_strict',ev/f'artifacts/exp024/eval_seed{s}_new96'),('fixed',ev/f'artifacts/exp026_eval/depfix_seed{s}_new96')]:
            g,v,source=load_bundle(p,ids);bs[arm]=(g,v);result['sources'][f'{arm}_{s}']=source
        baseline=bs['old_strict']
        drift={}
        for arm,(g,v) in bs.items():
            if any(g[q,i,'sft']['code_sha256']!=baseline[0][q,i,'sft']['code_sha256'] for q in ids for i in range(4)):
                raise ValueError('SFT reference content changed')
            drift[arm]=sum(v[q,i,'sft']['passed']!=baseline[1][q,i,'sft']['passed'] for q in ids for i in range(4))
        result['seeds'][str(s)]={'metrics':{a:summary(g,v,ids) for a,(g,v) in bs.items()},
            'fixed_minus_strict':compare(bs['old_strict'][1],bs['fixed'][1],ids),
            'fixed_minus_old_full':compare(bs['old_full'][1],bs['fixed'][1],ids),'sft_verdict_drift':drift,
            'same_code_as_strict':sum(bs['fixed'][0][q,i,'grpo']['code_sha256']==baseline[0][q,i,'grpo']['code_sha256'] for q in ids for i in range(4))}
    result['status']='complete_with_reference_drift' if any(any(x['sft_verdict_drift'].values()) for x in result['seeds'].values()) else 'complete'
    return result

def markdown(d):
    text=['# 2026-10-18｜EXP-026 同题配对复测完成','',
          '由已提交脚本校验冻结检查点后自动生成。只导出汇总，不导出候选代码、提示正文或隐藏测试。',
          '本阶段复测已经观察过的同一批96题，不是新的盲测。每模型384候选；SFT候选复用已校验的冻结答案。','',
          '| 种子 | 模型 | 正确候选/384 | 首样本/96 | 四样本覆盖/96 | NameError |',
          '|---|---|---:|---:|---:|---:|']
    for s,r in d['seeds'].items():
        for arm,m in r['metrics'].items():text.append(f'| {s} | {arm} | {m["correct"]} | {m["first"]} | {m["coverage"]} | {m["nameerror"]} |')
        z=r['fixed_minus_strict'];text.extend(['',f'种子{s}，修复版相对旧严格奖励：正确候选率差 {100*z["delta"][0]:+.4f} 个百分点；按题目配对bootstrap95%区间 [{100*z["ci95"][0][0]:+.4f}, {100*z["ci95"][1][0]:+.4f}]。错→对 {z["repair_pairs"]} 次，对→错 {z["regression_pairs"]} 次。',f'SFT判题漂移计数：{r["sft_verdict_drift"]}；修复版与旧严格奖励的代码文本相同 {r["same_code_as_strict"]}/384。',''])
    text.extend(['## 解释边界','只有两个训练种子，区间是逐种子的探索性区间，没有多重比较校正；不根据单个较好种子宣布稳定泛化。','缺失依赖分类修复是否改善总体效果，应与这里的实际配对结果区分。诊断反例通过不等于编程能力提升。','固定Docker增强test-field不是官方EvalPlus排行榜协议。任何SFT判题漂移均已显式报告，不静默排除任务。','',f'完整性状态：`{d["status"]}`。公开题目SHA256：`{d["public_sha256"]}`。'])
    return '\n'.join(text)+'\n'

def publish(text,out):
    endpoint=f'repos/{REPO}/contents/{REPORT}'
    q=subprocess.run(['gh','api',endpoint+'?ref='+BRANCH],capture_output=True,text=True)
    if q.returncode==0:
        existing=json.loads(q.stdout)
        if base64.b64decode(existing['content']).decode()!=text:raise ValueError('report already exists with different contents')
        return {'status':'already_published_identical','path':REPORT}
    if '404' not in q.stderr and 'Not Found' not in q.stdout:raise RuntimeError('GitHub report lookup failed; not writing')
    payload={'message':'results(EXP026): automatically finalize frozen paired evaluation','branch':BRANCH,'content':base64.b64encode(text.encode()).decode()}
    p=subprocess.run(['gh','api','-X','PUT',endpoint,'--input','-'],input=json.dumps(payload),capture_output=True,text=True,check=True)
    r=json.loads(p.stdout)
    return {'status':'published','commit_sha':r['commit']['sha'],'path':REPORT,'html_url':r['content']['html_url']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--train-root',type=Path,required=True);p.add_argument('--eval-root',type=Path,required=True);p.add_argument('--publish-github',action='store_true');a=p.parse_args()
    deadline=time.monotonic()+10800
    needed=[a.eval_root/f'artifacts/exp026_eval/depfix_seed{s}_new96/score_summary.json' for s in (43,44)]
    while not all(x.is_file() for x in needed):
        if time.monotonic()>deadline:raise TimeoutError('evaluation summaries not committed within bounded wait')
        time.sleep(15)
    out=a.train_root/'artifacts/exp026';d=analyze(a.train_root,a.eval_root);text=markdown(d)
    write_once(out/'paired_evaluation_final.json',d)
    with (out/'paired_evaluation_final.md').open('x') as f:f.write(text)
    if a.publish_github:write_once(out/'github_delivery.json',publish(text,out))
    print('EXP026_STATS_AND_REPORT_COMPLETE',flush=True)
if __name__=='__main__':main()
