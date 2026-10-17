"""Finalize frozen scores while separating result integrity from causal validity.

No raw programs or tests exported. Identical-code runtime drift must be audited
before publication. A complete experiment can still be a confounded ablation.
"""
from __future__ import annotations
import argparse,base64,json,subprocess,time
from pathlib import Path
import numpy as np
from coderl_lab.analysis.exp026_runner import load_bundle,rows,write_once,PUBLIC_SHA,sha
from coderl_lab.analysis.exp026_causal_gate import compare_trace

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
    boot=x[indices].mean(axis=1);ci=np.quantile(boot,[.025,.975],axis=0)
    return {'delta':x.mean(axis=0).tolist(),'ci95':ci.tolist(),'repair_pairs':up,'regression_pairs':down,
            'resample_unit':'task','bootstrap_seed':20261018,'replicates':20000,'warning':'nominal intervals; only two training seeds'}

def analyze(train,ev):
    public=ev/'artifacts/exp024/frozen_new96/public96_prompts.jsonl'
    if sha(public)!=PUBLIC_SHA:raise ValueError('cohort changed')
    ids=sorted(r['task_id'] for r in rows(public));result={'experiment':'EXP026-final-evaluation','seeds':{},'sources':{},'public_sha256':PUBLIC_SHA,'causal_validity':{}}
    for s in (43,44):
        traces=[]
        for path in (train/f'artifacts/exp024/strict_seed{s}',train/f'artifacts/exp026/depfix_seed{s}'):
            source=path/'reward_attempts.jsonl';m=json.loads((path/'summary.json').read_text())
            if sha(source)!=m['reward_attempt_log_sha256']:raise ValueError('training log mutation')
            traces.append(rows(source))
        result['causal_validity'][str(s)]=compare_trace(*traces)
        bs={};golden={}
        for arm,p in [('old_full',ev/f'artifacts/exp025/old_seed{s}_new96'),('old_strict',ev/f'artifacts/exp024/eval_seed{s}_new96'),('fixed',ev/f'artifacts/exp026_eval/depfix_seed{s}_new96')]:
            g,v,source=load_bundle(p,ids);bs[arm]=(g,v);result['sources'][f'{arm}_{s}']=source
            golden[arm]=json.loads((p/'score_summary.json').read_text())['reference_solution_test_status']
        baseline=bs['old_strict'];drift={}
        for arm,(g,v) in bs.items():
            if any(g[q,i,'sft']['code_sha256']!=baseline[0][q,i,'sft']['code_sha256'] for q in ids for i in range(4)):
                raise ValueError('SFT reference content changed')
            drift[arm]=sum(v[q,i,'sft']['passed']!=baseline[1][q,i,'sft']['passed'] for q in ids for i in range(4))
        result['seeds'][str(s)]={'metrics':{a:summary(g,v,ids) for a,(g,v) in bs.items()},
            'fixed_minus_strict':compare(bs['old_strict'][1],bs['fixed'][1],ids),
            'fixed_minus_old_full':compare(bs['old_full'][1],bs['fixed'][1],ids),'sft_verdict_drift':drift,'golden_controls':golden,
            'same_code_as_strict':sum(bs['fixed'][0][q,i,'grpo']['code_sha256']==baseline[0][q,i,'grpo']['code_sha256'] for q in ids for i in range(4))}
    result['status']='complete_with_reference_drift' if any(any(x['sft_verdict_drift'].values()) for x in result['seeds'].values()) else 'complete'
    result['causal_status']='runtime_confounded' if any(x['status']=='runtime_confounded' for x in result['causal_validity'].values()) else 'no_observed_same_code_execution_drift'
    return result

def markdown(d):
    if 'causal_status' not in d or 'causal_validity' not in d:raise ValueError('missing causal validity audit; cannot publish')
    text=['# 2026-10-18｜EXP-026 同题配对复测完成','',
          '由已提交脚本校验冻结检查点后自动生成。只导出汇总，不导出候选代码、提示正文或隐藏测试。',
          '本阶段复测已经观察过的同一批96题，不是新的盲测。每模型384候选；SFT候选复用已校验的冻结答案。','',
          f'**因果归因状态：`{d["causal_status"]}`。完成评分不等于完成了无混杂的奖励消融。**','']
    if d['causal_status']=='runtime_confounded':
        text.extend(['seed43存在相同代码在历史训练中得到正奖励、此次却超时记零的变化，发生在依赖分类变化之前。因此不得将seed43的模型或分数变化单独归因于依赖奖励修复。执行时限、运行条件与奖励分类的影响尚未分离。',''])
    text.extend(['| 种子 | 模型 | 正确候选/384 | 首样本/96 | 四样本覆盖/96 | NameError |','|---|---|---:|---:|---:|---:|'])
    for s,r in d['seeds'].items():
        for arm,m in r['metrics'].items():text.append(f'| {s} | {arm} | {m["correct"]} | {m["first"]} | {m["coverage"]} | {m["nameerror"]} |')
    for s,r in d['seeds'].items():
        z=r['fixed_minus_strict'];gate=d['causal_validity'][s]
        text.extend(['',f'种子{s}，修复版相对旧严格奖励：正确候选率差 {100*z["delta"][0]:+.4f} 个百分点；按题目配对bootstrap95%区间 [{100*z["ci95"][0][0]:+.4f}, {100*z["ci95"][1][0]:+.4f}]。错→对 {z["repair_pairs"]} 次，对→错 {z["regression_pairs"]} 次。',
                     f'SFT判题漂移：{r["sft_verdict_drift"]}；与旧严格奖励代码文本相同 {r["same_code_as_strict"]}/384；参考解结果：{r["golden_controls"]}。',
                     f'归因诊断：{gate["status"]}；相同代码的执行结果漂移 {len(gate["same_code_probe_drift"])} 次；同一probe上分类修正影响 {gate["classification_effect_count"]} 个候选；首次生成代码分歧索引 {gate["first_generation_code_difference"]}。'])
    text.extend(['','## 解释边界','只有两个训练种子，区间是逐种子的探索性区间，没有多重比较校正；不根据单个较好种子宣布稳定泛化。','修复真实的异常分类缺陷，不证明该修复导致模型能力增益。并发负载、温度或其他环境因素可能影响墙钟超时，但当前不认定具体根因。','固定Docker增强test-field不是官方EvalPlus排行榜协议。任何SFT判题漂移均显式报告，不静默排除任务。','',f'数据完整性状态：`{d["status"]}`；因果状态：`{d["causal_status"]}`。公开题目SHA256：`{d["public_sha256"]}`。'])
    return '\n'.join(text)+'\n'

def publish(text,out):
    endpoint=f'repos/{REPO}/contents/{REPORT}'
    q=subprocess.run(['gh','api',endpoint+'?ref='+BRANCH],capture_output=True,text=True,timeout=30)
    if q.returncode==0:
        existing=json.loads(q.stdout)
        if base64.b64decode(existing['content']).decode()!=text:raise ValueError('report already exists with different contents')
        return {'status':'already_published_identical','path':REPORT}
    if '404' not in q.stderr and 'Not Found' not in q.stdout:raise RuntimeError('GitHub report lookup failed; not writing')
    payload={'message':'results(EXP026): finalize paired scores with explicit runtime-confounding audit','branch':BRANCH,'content':base64.b64encode(text.encode()).decode()}
    p=subprocess.run(['gh','api','-X','PUT',endpoint,'--input','-'],input=json.dumps(payload),capture_output=True,text=True,check=True,timeout=30)
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
