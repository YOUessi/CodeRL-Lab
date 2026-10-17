"""Tang-only EXP028: no manual phase handoffs, no unbounded retries or budget growth.
Only safe summaries may be published. All candidates and test bodies stay local.
"""
from __future__ import annotations
import argparse,base64,datetime,fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
import yaml
from coderl_lab.analysis import exp027_matched_eval as ev
from coderl_lab.analysis.exp028_protocol import (ARMS,SEEDS,STEPS,SFT_SHA,canonical,sha,rows,once,public_rows,validate_generations)
from coderl_lab.train.exp027_matched_reward import protocol

TRAIN=Path('/home/you/projects/CodeRL-Lab-track-train017')
SFT=Path('/home/you/projects/CodeRL-Lab-track-train011/artifacts/train011/qlora-ultrachat')
BRANCH='exp/training-coverage-20261018'
REPORT='docs/experiment-logs/2026-10-18-exp028-coverage-result.md'


def state(root,stage,status,**extra):
    x={'experiment':'EXP028','stage':stage,'status':status,'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra}
    tmp=root/'state.tmp';tmp.write_text(canonical(x)+'\n');os.replace(tmp,root/'state.json')
    with (root/'events.jsonl').open('a') as f:f.write(canonical(x)+'\n')
    print('STATE',canonical(x),flush=True)


def child(cmd,work,env,log,gpu=False):
    with Path(log).open('a') as f:
        if gpu:
            with open('/tmp/coderl_lab_gpu_experiment_lock','a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                subprocess.run([sys.executable,'-m','coderl_lab.train.gpu_preflight','--min-free-mib','12000'],cwd=work,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
                subprocess.run(cmd,cwd=work,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        else:subprocess.run(cmd,cwd=work,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)


def verify_data(root):
    data=root/'data';m=json.loads((data/'manifest.json').read_text())
    for path,h in m['files'].items():
        if sha(data/path)!=h:raise ValueError('training data drift')
    ex=root/'external96';external=json.loads((ex/'manifest.json').read_text())
    if sha(ex/'public.jsonl')!=external['public_sha256']:raise ValueError('public evaluation drift')
    if external['selected']!=96 or external['excluded_task_count']!=216:raise ValueError('wrong external cohort')
    return m,external


def make_config(root,arm,seed):
    c=yaml.safe_load((TRAIN/f'configs/exp024_strict_seed{seed}.yaml').read_text())
    m=json.loads((root/'data'/arm/'manifest.json').read_text())
    c['experiment']['coverage_arm']=arm
    c['reward']['wall_guard_seconds']=60.
    c['reward']['timeout_seconds']=60
    c['reward']['active_definition']='EXP027 corrected 0.1E+0.1D+0.8S'
    c['dataset'].update({'train_tasks':str(root/'data'/arm/'train.jsonl'),'train_sha256':m['sha256']['train'],
                         'heldout_public_sha256':m['sha256']['holdout_public'],'heldout_oracle_sha256':m['sha256']['holdout_oracle']})
    c['runs']['formal_output']=str(root/f'{arm}_seed{seed}')
    c['runs']['smoke_output']=str(root/f'{arm}_smoke_seed{seed}')
    path=root/f'{arm}_seed{seed}_config.yaml';text=yaml.safe_dump(c,sort_keys=False)
    if path.exists():
        if path.read_text()!=text:raise ValueError('config drift')
    else:path.write_text(text)
    return path


def verify_model(root,arm,seed):
    p=root/f'{arm}_seed{seed}';s=json.loads((p/'summary.json').read_text())
    rr=rows(p/'reward_attempts.jsonl')
    if (s['status']!='real_GPU_GRPO_completed' or s['global_step']!=32 or s['coverage_arm']!=arm or s['training_seed']!=seed
        or s['saved_adapter_sha256']!=sha(p/'adapter_model.safetensors')
        or s['warm_start_source_sft_adapter_sha256']!=SFT_SHA
        or s['train_sha256']!=sha(root/'data'/arm/'train.jsonl')
        or s['reward_attempt_log_sha256']!=sha(p/'reward_attempts.jsonl') or len(rr)!=128):
        raise ValueError('training proof mismatch')
    if (p/'unscorable_abort.json').exists():raise ValueError('unscorable training arm')
    if any(r['variant']!='corrected' for r in rr):raise ValueError('reward changed')
    groups=[rr[i:i+4] for i in range(0,len(rr),4)]
    if any(len({r['task_id'] for r in g})!=1 for g in groups):raise ValueError('invalid rollout group')
    return {'arm':arm,'seed':seed,'optimizer_steps':32,'reward_requests':128,'adapter_sha256':s['saved_adapter_sha256'],
            'changed_elements':s['initial_SFT_parameter_elements_changed_after_grpo'],
            'mixed_groups':sum(len({r['reward'] for r in g})>1 for g in groups),
            'runtime_seconds':s['train_runtime_seconds'],'reward_source_sha256':s['active_reward_source_sha256'],
            'training_loop_sha256':s['active_training_loop_sha256'],'config_sha256':sha(root/f'{arm}_seed{seed}_config.yaml')}


def integration(root):
    if (root/'evaluation_integration.json').exists():return
    good=ev.execute('def f(x): return x+1','assert f(2)==3','f',cpu_seconds=1,wall_seconds=10)
    budget=ev.execute('def f(x):\n while True: pass','f(0)','f',cpu_seconds=1,wall_seconds=10)
    if not good['passed'] or budget['reason']!='cpu_budget_exceeded':raise ValueError('real scoring integration failed')
    once(root/'evaluation_integration.json',{'correct':good,'resource':budget,'protocol':ev.protocol()})


def evaluate(root,work,env,runproof,external):
    integration(root)
    models={'sft':{'path':SFT,'sha':SFT_SHA}}
    for r in runproof:models[f"{r['arm']}_{r['seed']}"]={'path':root/f"{r['arm']}_seed{r['seed']}",'sha':r['adapter_sha256']}
    public=public_rows(root/'external96/public.jsonl');ids={x['task_id'] for x in public}
    if len(ids)!=96:raise ValueError('evaluation size changed')
    unique={};sources={}
    for name,model in models.items():
        w=model['sha']
        if w in unique:continue
        out=root/'evaluation'/('generation_'+w[:16]);out.mkdir(parents=True,exist_ok=True)
        state(root,'evaluation_generation','running',model=name)
        if not (out/'summary.json').exists():
            child([sys.executable,'-m','coderl_lab.analysis.exp028_generate','--public',str(root/'external96/public.jsonl'),
                   '--public-sha',external['public_sha256'],'--adapter',str(model['path']),'--adapter-sha',w,'--out',str(out)],
                  work,env,root/f'generate_{name}.log',gpu=True)
        s=json.loads((out/'summary.json').read_text())
        rr=rows(out/'checkpoint.jsonl');gg,expected=validate_generations(rr,public)
        if set(gg)!=expected or s['checkpoint_sha256']!=sha(out/'checkpoint.jsonl') or s['adapter_sha256']!=w or s['public_sha256']!=external['public_sha256']:
            raise ValueError('unfrozen generation')
        unique[w]=rr;sources[w]={'generation_sha256':s['checkpoint_sha256'],'count':384,'generated_tokens':s['generated_tokens']}
    once(root/'evaluation/generation_freeze.json',{'status':'all_models_generated_frozen','models':{n:m['sha'] for n,m in models.items()},
                                                  'sources':sources,'public_sha256':external['public_sha256'],'protocol':ev.protocol()})
    # First external hidden-test access: all model generations are complete.
    private=root/'external96/private.jsonl'
    if sha(private)!=external['private_sha256']:raise ValueError('hidden tests drift')
    oracle={r['task_id']:r for r in rows(private)}
    if set(oracle)!=ids:raise ValueError('oracle task set mismatch')
    cache=root/'evaluation/score_cache';gp=root/'evaluation/golden.jsonl'
    golden={r['task_id']:r for r in rows(gp)} if gp.exists() else {}
    state(root,'golden_controls','running')
    for q in sorted(ids):
        if q not in golden:
            v=ev.score_cached(cache,oracle[q]['canonical_code'],oracle[q]);record={'task_id':q,**v}
            with gp.open('a') as f:f.write(canonical(record)+'\n');f.flush()
            golden[q]=record
    if len(golden)!=96:raise ValueError('golden incomplete')
    if not all(r['passed'] for r in golden.values()):
        return {'experiment':'EXP028','status':'stopped_reference_controls','golden_passed':sum(x['passed'] for x in golden.values()),
                'golden_total':96,'tasks_dropped':0,'training':runproof,'gate':json.loads((root/'gate_report.json').read_text())}
    judged={}
    for w,rr in unique.items():
        state(root,'candidate_scoring','running',model_sha=w)
        path=root/'evaluation'/('scored_'+w[:16]+'.jsonl')
        done={(r['task_id'],r['sample_id']):r for r in rows(path)} if path.exists() else {}
        for r in rr:
            key=r['task_id'],r['sample_id']
            if key in done:
                if done[key]['code_sha256']!=r['code_sha256']:raise ValueError('score checkpoint changed')
                continue
            v=ev.score_cached(cache,r['raw_code'],oracle[key[0]])
            record={'task_id':key[0],'sample_id':key[1],'code_sha256':r['code_sha256'],**v}
            with path.open('a') as f:f.write(canonical(record)+'\n');f.flush()
            done[key]=record
        if len(done)!=384:raise ValueError('incomplete model scoring')
        judged[w]=list(done.values())
    final={'experiment':'EXP028','status':'complete','gate':json.loads((root/'gate_report.json').read_text()),'training':runproof,
           'golden_passed':96,'golden_total':96,'public_sha256':external['public_sha256'],'evaluation_protocol':ev.protocol(),
           'fresh_evaluation_candidates':len(unique)*384,'metrics':{n:ev.metrics(judged[m['sha']]) for n,m in models.items()},'comparisons':{}}
    for seed in SEEDS:
        a=judged[models[f'narrow_{seed}']['sha']];b=judged[models[f'broad_{seed}']['sha']]
        final['comparisons'][str(seed)]=ev.compare(ev.matrix(a),ev.matrix(b))
    final['limitation']='Exploratory two-seed fixed-budget family-coverage study. Same MBPP+ benchmark, not cross-benchmark transfer; no pretrained contamination guarantee. Actual compute and tokens can differ.'
    return final


def render(final):
    text=['# 2026-10-18｜EXP-028 训练题型覆盖实验','',f"状态：`{final['status']}`。仅Tang执行，固定预算，不对不利结果追加种子或改门槛。",'']
    g=final.get('gate',{})
    if g:
        text+=['## 训练前检查',f"通过：{g.get('passed')}；停止原因：{g.get('stop_reasons',[])}。"]
        for arm,r in g.get('arms',{}).items():text.append(f"{arm}：奖励有差异 {r['mixed_groups']}/48 题；新增族有差异 {r['new_mixed_groups']} 题；语法入口通过 {r['syntax_fraction']:.3%}；达到token上限 {r['clip_fraction']:.3%}。")
    if final.get('metrics'):
        text+=['','## 独立冻结96题结果','|模型|正确候选/384|首样本/96|四样本覆盖/96|NameError|','|---|---:|---:|---:|---:|']
        for n,m in final['metrics'].items():text.append(f"|{n}|{m['correct']}|{m['first']}|{m['coverage']}|{m['reasons'].get('NameError',0)}|")
        for s,c in final['comparisons'].items():text.append(f"seed{s} broad−narrow：正确候选率 {c['candidate_delta']*100:+.4f} 个百分点，按题配对95%区间 [{c['ci95'][0][0]*100:+.4f}, {c['ci95'][1][0]*100:+.4f}]；修复{c['repairs']}，回退{c['regressions']}。")
    text+=['','## 边界','训练集各48条，narrow为6族各8条，broad为12族各4条；奖励、执行协议、32步、4候选、256token、seed43/44固定。不是所有输入接口同时扩展。',
           '新96题与旧216题不交叉，但仍来自MBPP+，不是跨基准验证。原始题目、代码和隐藏测试不公开上传。两种子区间为探索性分析；如未见一致收益就接受未确认结果。']
    return '\n'.join(text)+'\n'


def publish(root,text):
    endpoint='repos/YOUessi/CodeRL-Lab/contents/'+REPORT
    q=subprocess.run(['gh','api',endpoint+'?ref='+BRANCH],capture_output=True,text=True,timeout=30)
    if q.returncode==0:
        if base64.b64decode(json.loads(q.stdout)['content']).decode()!=text:raise ValueError('report already exists with different result')
        return {'status':'already_published','path':REPORT}
    if '404' not in q.stderr and 'Not Found' not in q.stdout:raise RuntimeError('GitHub read failed; no write')
    payload={'message':'results(EXP028): complete bounded coverage stage and record gate/outcomes','branch':BRANCH,'content':base64.b64encode(text.encode()).decode()}
    p=subprocess.run(['gh','api','-X','PUT',endpoint,'--input','-'],input=json.dumps(payload),capture_output=True,text=True,timeout=30,check=True)
    r=json.loads(p.stdout);return {'status':'published','path':REPORT,'commit_sha':r['commit']['sha']}


def run(root,work,attach_gate,publish_github):
    root.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(work/'src'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    files=['src/coderl_lab/analysis/exp028_protocol.py','src/coderl_lab/analysis/exp028_generate.py','src/coderl_lab/train/exp028_7b_grpo.py',
           'src/coderl_lab/train/exp024_7b_grpo.py','src/coderl_lab/train/exp027_matched_reward.py','src/coderl_lab/analysis/exp027_matched_eval.py','scripts/run_exp028_chain.py']
    once(root/'active_code.json',{'files':{f:sha(work/f) for f in files},'reward_protocol':protocol()})
    _,external=verify_data(root)
    state(root,'pretraining_gate','waiting' if attach_gate else 'running')
    if not (root/'gate_report.json').exists():
        if attach_gate:
            deadline=time.monotonic()+7200
            while not (root/'gate_report.json').exists():
                if (root/'gate_failure.json').exists():raise RuntimeError('pretraining gate execution failed')
                if time.monotonic()>deadline:raise TimeoutError('bounded gate wait expired')
                time.sleep(10)
        else:child([sys.executable,str(work/'scripts/exp028_pretrain_gate.py'),'--root',str(root)],work,env,root/'gate_chain.log')
    g=json.loads((root/'gate_report.json').read_text())
    if g['score_sha256']!=sha(root/'pretrain_scored.jsonl') or g['data_manifest_sha256']!=sha(root/'data/manifest.json'):raise ValueError('gate proof changed')
    if not g['passed']:
        final={'experiment':'EXP028','status':'stopped_by_preregistered_gate','gate':g,'new_training_runs':0}
    else:
        proofs=[]
        for seed in SEEDS:
            for arm in ARMS:
                state(root,'training','running',arm=arm,seed=seed)
                cfg=make_config(root,arm,seed);out=root/f'{arm}_seed{seed}'
                if not (out/'summary.json').exists():
                    if (out/'identity.json').exists():raise RuntimeError('partial training detected; no unregistered restart')
                    child([sys.executable,'-m','coderl_lab.train.exp028_7b_grpo','--coverage',arm,'--cache-root',str(root/'training_observations'),
                           '--gate-report',str(root/'gate_report.json'),'--arm','strict','--config',str(cfg),'--data-dir',str(root/'data'/arm),
                           '--sft-adapter',str(SFT),'--output-dir',str(out),'--max-steps','32','--run-kind','formal'],work,env,root/f'train_{arm}_{seed}.log',gpu=True)
                proofs.append(verify_model(root,arm,seed))
        if len({(x['reward_source_sha256'],x['training_loop_sha256']) for x in proofs})!=1:raise ValueError('training mechanisms differ')
        once(root/'four_trainings.json',proofs)
        final=evaluate(root,work,env,proofs,external)
    once(root/'final.json',final);text=render(final)
    p=root/'final.md'
    if not p.exists():p.write_text(text)
    elif p.read_text()!=text:raise ValueError('final report changed')
    if publish_github:once(root/'github_delivery.json',publish(root,text))
    state(root,'finished',final['status'],final_sha256=sha(root/'final.json'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--attach-gate',action='store_true');p.add_argument('--publish-github',action='store_true');a=p.parse_args()
    if os.uname().nodename!='Tang':raise RuntimeError('Tang only')
    work=Path(__file__).resolve().parents[1]
    with open('/tmp/coderl_exp028_chain.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a.root,work,a.attach_gate,a.publish_github)
        except Exception as e:
            state(a.root,'failed','failed',error_type=type(e).__name__,reason=str(e))
            once(a.root/'chain_failure.json',{'error_type':type(e).__name__,'reason':str(e),'no_automatic_retry':True})
            raise
if __name__=='__main__':main()
