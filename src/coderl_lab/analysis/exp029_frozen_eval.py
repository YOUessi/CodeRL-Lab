"""EXP029: evaluation-only, uniform 4GiB memory for references and candidates.

Old EXP028 artifacts are read-only. No model training, new generation, task
replacement, hidden-test modification or reuse of historical verdicts occurs.
The resource revision was specified from a reference failure before model scores.
"""
from __future__ import annotations
import argparse
import collections
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
from coderl_lab.analysis import exp027_matched_eval as ev
from coderl_lab.analysis import exp028_reference_controls as controls
from coderl_lab.analysis.exp028_protocol import canonical,rows,sha,once
from coderl_lab.train.exp027_guarded_probe import SandboxIndeterminate

MEMORY_BYTES=4*1024**3
MIN_HOST_AVAILABLE=8*1024**3
MODEL_NAMES={'sft','narrow_43','broad_43','narrow_44','broad_44'}


def protocol():
    return {**ev.protocol(),'experiment':'EXP029-uniform-memory4g-evaluation',
            'memory_bytes':MEMORY_BYTES,'swap_bytes':0,'cpu_hard_seconds':16,
            'different_from_historical_1GiB':True,'max_concurrent_containers':1,
            'unconfirmed_exit':'abort_no_score_no_automatic_retry',
            'memory_failure':'engine_OOMKilled_and_stopped_and_exit137',
            'source_sha256':sha(__file__)}


def ensure_host_capacity(meminfo=Path('/proc/meminfo')):
    lines=Path(meminfo).read_text().splitlines()
    values=[int(x.split()[1])*1024 for x in lines if x.startswith('MemAvailable:')]
    if len(values)!=1 or values[0]<MIN_HOST_AVAILABLE:
        raise SandboxIndeterminate('insufficient_host_memory_no_execution')


def control_gate(records,ids):
    if len(set(ids))!=96:raise ValueError('exactly96 controls required')
    return controls.control_summary(records,ids)


def _fingerprint(code,test,entry):
    return {'code_sha256':hashlib.sha256(code.encode()).hexdigest(),
            'test_sha256':hashlib.sha256(test.encode()).hexdigest(),
            'entry_point':entry,'protocol':protocol()}


def _read_complete(path,fp,key):
    r=json.loads(path.read_text());body={k:v for k,v in r.items() if k!='record_sha256'}
    if (r.get('record_sha256')!=ev.objsha(body) or r.get('key')!=key
        or any(r.get(k)!=v for k,v in fp.items()) or r.get('cleanup_confirmed') is not True
        or type(r.get('passed')) is not bool or r['passed']!=(r.get('reason')=='none')):
        raise ValueError('immutable scoring evidence changed')
    return r


def score(raw,oracle,cache):
    """One isolated execution or verified exact-code/test/protocol cache hit."""
    if not isinstance(raw,str):raise ValueError('candidate source must be text')
    code,normalizer=ev.extract_code(raw)
    test=oracle['test'];entry=oracle['entry_point']
    if not isinstance(test,str) or not isinstance(entry,str):raise ValueError('invalid oracle')
    fp=_fingerprint(code,test,entry);key=ev.objsha(fp)
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    prefix=cache/key;complete=prefix.with_suffix('.complete.json')
    if complete.exists():
        return {**_read_complete(complete,fp,key),'cache_hit':True,'normalizer':normalizer}
    prepared=prefix.with_suffix('.prepared.json')
    if prepared.exists():raise RuntimeError('unfinished scoring attempt; preserve evidence, no retry')
    syntax,entry_valid=ev.ast_valid(code,entry)
    r={**fp,'key':key,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'container_executed':False,'syntax_and_entry':bool(syntax and entry_valid)}
    once(prepared,r)
    if not syntax or not entry_valid:
        r.update(passed=False,reason='invalid_syntax_or_entry',resource_limited=False,
                 cleanup_confirmed=True,elapsed_seconds=0.)
    else:
        name='coderl-eval029-'+uuid.uuid4().hex[:16]
        start=time.monotonic();error=None;verdict=None
        cmd=['docker','run','--name',name,'-i','--network','none','--memory','4g','--memory-swap','4g',
             '--cpus','1','--pids-limit','64','--read-only','--cap-drop','ALL',
             '--security-opt','no-new-privileges','--user','65534:65534',
             '--tmpfs','/tmp:rw,nosuid,size=128m','-e','PYTHONDONTWRITEBYTECODE=1',
             '-e','OPENBLAS_NUM_THREADS=1','--entrypoint','/usr/bin/python3',ev.IMAGE_ID,'-c',ev.DRIVER]
        try:
            ensure_host_capacity()
            r['container_executed']=True
            try:
                p=subprocess.run(cmd,input=json.dumps({'code':code,'test':test,'entry_point':entry}),
                                 capture_output=True,text=True,timeout=60)
            except subprocess.TimeoutExpired as exc:
                raise SandboxIndeterminate('host_wall_guard_no_score') from exc
            q=subprocess.run(['docker','inspect','--format','{{json .State}}',name],
                             capture_output=True,text=True,check=True,timeout=10)
            s=json.loads(q.stdout)
            r.update(exit_code=p.returncode,
                     container_state={k:s.get(k) for k in ('Running','OOMKilled','ExitCode','Dead','Error')},
                     cpu_budget_marker_count=p.stdout.splitlines().count('__EXP027_CPU_BUDGET__'))
            once(prefix.with_suffix('.engine.json'),r)
            verdict=controls.classify_exit(p.returncode,p.stdout,s)
        except Exception as exc:
            error=exc
        finally:
            try:
                c=subprocess.run(['docker','rm','-f',name],capture_output=True,text=True,timeout=15)
                if c.returncode and 'No such container' not in c.stderr:
                    raise SandboxIndeterminate('container_cleanup_failed')
                r['cleanup_confirmed']=True
            except (OSError,subprocess.SubprocessError,SandboxIndeterminate) as exc:
                r['cleanup_confirmed']=False
                error=SandboxIndeterminate('container_cleanup_unconfirmed_'+type(exc).__name__)
        r['elapsed_seconds']=round(time.monotonic()-start,4)
        if error is not None:
            once(prefix.with_suffix('.failed.json'),{**r,'error_type':type(error).__name__,
                 'reason':str(error),'score_assigned':False})
            raise error
        r.update(verdict)
    r['record_sha256']=ev.objsha(r)
    once(complete,r)
    return {**r,'cache_hit':False,'normalizer':normalizer}


def _state(out,stage,**extra):
    data={'experiment':'EXP029','stage':stage,'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra}
    p=out/'state.tmp';p.write_text(canonical(data)+'\n');os.replace(p,out/'state.json')
    print('STATE',canonical(data),flush=True)


def _append(path,row):
    with path.open('a') as f:
        f.write(canonical(row)+'\n');f.flush();os.fsync(f.fileno())


def _brief(r):
    return {k:r[k] for k in ('passed','reason','resource_limited','key','record_sha256',
                             'cache_hit','container_executed','normalizer')}


def render(final):
    t=['# 2026-10-18｜EXP-029 冻结覆盖模型的统一4GiB评测','',
       '本轮仅修订评测资源，不重新训练、不重新生成、不删题、不修改测试代码。EXP028的1GiB参考失败与停止记录保留。',
       f"状态：{final['status']}；所有参考与候选统一4GiB/no-swap、1CPU、CPU软15/硬16秒、宿主墙钟60秒。",'',
       f"参考解：{final['reference_controls']['passed']}/96通过；全题集与五个模型的1920份生成均在本轮评分前冻结并校验。"]
    if final.get('metrics'):
        t+=['','|模型|正确候选/384|固定首样本/96|四次覆盖/96|NameError|资源失败|',
            '|---|---:|---:|---:|---:|---:|']
        for name in ('sft','narrow_43','broad_43','narrow_44','broad_44'):
            x=final['metrics'][name];resources=sum(x['reasons'].get(k,0) for k in ('cpu_budget_exceeded','memory_budget_exceeded'))
            t.append(f"|{name}|{x['correct']}|{x['first']}|{x['coverage']}|{x['reasons'].get('NameError',0)}|{resources}|")
        for seed,c in final['broad_minus_narrow'].items():
            t.append(f"seed{seed} broad−narrow：候选正确率差{c['candidate_delta']*100:+.4f}个百分点；题目配对bootstrap95%区间[{c['ci95'][0][0]*100:+.4f},{c['ci95'][1][0]*100:+.4f}]；修复{c['repairs']}对，回退{c['regressions']}对。")
        t+=['',f"逻辑候选评分{final['candidate_positions_scored']}；本轮真实独立容器观测{final['physical_container_executions']}（含参考解）；无新训练或生成。"]
    t+=['','## 解释边界',
        '内存由1GiB改为4GiB是看到参考资源失败后的明确协议修订，不是原协议的成功重跑。资源根据参考程序而非模型成绩决定；同一预算应用到所有模型。',
        '原有组为6族各8题，宽覆盖组为12族各4题，均48条、32步、4候选、256token；覆盖范围增加同时减少每族训练密度，是固定预算分配的组合干预。',
        '该96题仍来自MBPP+，非跨基准验证；两训练seed不足以精确刻画训练随机性。区间为逐种子探索性区间，按题目抽样而非把384候选看作384独立题。',
        '候选正确比例用于估计单次采样成功率；固定sample0与四次覆盖另列。相同程序执行去重只节省计算，不增加独立样本数。',
        '不因某个较好种子追加训练或筛选有利任务，不将参考解通过当模型正确率。原代码与测试只留Tang。']
    return '\n'.join(t)+'\n'


def run(root,out):
    ids,oracle,source=controls.validate_frozen_run(root)
    freeze=json.loads((root/'evaluation/generation_freeze.json').read_text())
    if set(freeze['models'])!=MODEL_NAMES:raise ValueError('unexpected model list')
    identity={'experiment':'EXP029','source':source,'protocol':protocol(),
              'original_closure_sha256':sha(root/'control_validation_closure.json'),
              'fresh_training_runs':0,'fresh_generations':0,'reference_gate_before_model_scoring':True}
    once(out/'identity.json',identity)
    if (out/'final.json').exists():
        final=json.loads((out/'final.json').read_text())
        if final['identity_sha256']!=sha(out/'identity.json'):raise ValueError('result provenance drift')
        return final
    cache=out/'observations';refpath=out/'reference_results.jsonl'
    existing=rows(refpath) if refpath.exists() else []
    if len({x['task_id'] for x in existing})!=len(existing):raise ValueError('duplicate reference results')
    for x in existing:
        if x['task_id'] not in ids:raise ValueError('unexpected reference task')
    done={x['task_id']:x for x in existing}
    _state(out,'reference_controls',completed=len(done),total=96)
    for q in sorted(ids):
        r=score(oracle[q]['canonical_code'],oracle[q],cache)
        record={'task_id':q,**_brief(r)}
        if q in done:
            if done[q]['record_sha256']!=record['record_sha256']:raise ValueError('reference record drift')
        else:
            _append(refpath,record);done[q]=record
            print('REFERENCE',len(done),'/96',q,r['reason'],flush=True)
    gate=control_gate(list(done.values()),ids)
    once(out/'reference_gate.json',gate)
    if not gate['model_scoring_permitted']:
        final={**identity,'identity_sha256':sha(out/'identity.json'),'status':'stopped_reference_controls',
               'reference_controls':gate,'candidate_positions_scored':0}
        once(out/'final.json',final);once(out/'final_text.json',{'text':render(final)})
        _state(out,'stopped_reference_controls',failed_tasks=gate['failed_task_ids'])
        return final
    _state(out,'scoring_candidates',total=1920)
    judged={};score_sources={};changed_pairs={}
    for weight in sorted(freeze['sources']):
        genpath=root/'evaluation'/('generation_'+weight[:16])/'checkpoint.jsonl'
        if sha(genpath)!=freeze['sources'][weight]['generation_sha256']:raise ValueError('generation changed before scoring')
        gg=rows(genpath);sp=out/('scored_'+weight[:16]+'.jsonl')
        old=rows(sp) if sp.exists() else []
        scored={(x['task_id'],x['sample_id']):x for x in old}
        if len(old)!=len(scored):raise ValueError('duplicate scores')
        expected={(x['task_id'],x['sample_id']) for x in gg}
        if not set(scored)<=expected:raise ValueError('unknown score keys')
        for g in sorted(gg,key=lambda r:(r['task_id'],r['sample_id'])):
            k=g['task_id'],g['sample_id'];r=score(g['raw_code'],oracle[k[0]],cache)
            item={'task_id':k[0],'sample_id':k[1],'code_sha256':g['code_sha256'],**_brief(r)}
            if k in scored:
                if scored[k]['record_sha256']!=r['record_sha256'] or scored[k]['code_sha256']!=g['code_sha256']:
                    raise ValueError('score checkpoint drift')
            else:
                _append(sp,item);scored[k]=item
                if len(scored)%64==0:print('MODEL',weight[:12],len(scored),'/384',flush=True)
        if set(scored)!=expected or len(scored)!=384:raise ValueError('incomplete model scores')
        judged[weight]=list(scored.values());score_sources[weight]=sha(sp)
    metrics={name:ev.metrics(judged[w]) for name,w in freeze['models'].items()}
    comparisons={};versus_sft={}
    for seed in (43,44):
        a=judged[freeze['models'][f'narrow_{seed}']];b=judged[freeze['models'][f'broad_{seed}']]
        comparisons[str(seed)]=ev.compare(ev.matrix(a),ev.matrix(b))
        aa={(x['task_id'],x['sample_id']):x for x in a};bb={(x['task_id'],x['sample_id']):x for x in b}
        changed_pairs[str(seed)]=[{'task_id':k[0],'sample_id':k[1],
                    'direction':'repair' if bb[k]['passed'] else 'regression',
                    'narrow_reason':aa[k]['reason'],'broad_reason':bb[k]['reason']}
                 for k in sorted(aa) if aa[k]['passed']!=bb[k]['passed']]
        for arm in ('narrow','broad'):
            versus_sft[f'{arm}_{seed}']=ev.compare(ev.matrix(judged[freeze['models']['sft']]),
                                                  ev.matrix(judged[freeze['models'][f'{arm}_{seed}']]))
    evidence=[json.loads(p.read_text()) for p in cache.glob('*.complete.json')]
    final={**identity,'identity_sha256':sha(out/'identity.json'),'status':'complete',
           'reference_controls':gate,'reference_scores_sha256':sha(refpath),'metrics':metrics,
           'broad_minus_narrow':comparisons,'versus_sft':versus_sft,
           'paired_flips':changed_pairs,'scoring_files_sha256':score_sources,
           'physical_container_executions':sum(x['container_executed'] for x in evidence),
           'unique_observations_including_static_rejections':len(evidence),
           'candidate_positions_scored':sum(len(x) for x in judged.values()),
           'stopping_rule':'no_extra_seeds_no_reweighting_no_task_selection',
           'limit':'Post-reference-failure uniform resource revision; exploratory two-seed matched MBPP+ test, not official EvalPlus.'}
    once(out/'final.json',final)
    report=render(final);p=out/'final.md'
    if p.exists() and p.read_text()!=report:raise ValueError('report already exists with different content')
    if not p.exists():p.write_text(report)
    _state(out,'complete',final_sha256=sha(out/'final.json'))
    return final


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if os.uname().nodename!='Tang':raise RuntimeError('Tang only')
    root=a.root.resolve();out=a.out.resolve()
    if root.name!='exp028_v3' or out.name!='exp029_eval4g' or root.parent!=out.parent:
        raise ValueError('distinct registered artifact directories required')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'failure.json').exists():raise RuntimeError('explicit investigation required after failure')
    with open('/tmp/coderl_exp029_eval.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(root,out)
        except Exception as exc:
            once(out/'failure.json',{'error_type':type(exc).__name__,'reason':str(exc),
                                    'no_automatic_retry':True,'old_artifacts_modified':False})
            _state(out,'failed',error_type=type(exc).__name__,reason=str(exc));raise
if __name__=='__main__':main()
