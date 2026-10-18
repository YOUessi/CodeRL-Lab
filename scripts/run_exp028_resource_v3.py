"""Registered v3 recovery, not an in-place restart of failed v2.

Reuse frozen data and SFT gate generations only; redo gate scoring and all four
trainings under the same CPU/memory protocol. No threshold edits or resampling.
"""
from __future__ import annotations
import argparse,json,os,shutil,sys
from pathlib import Path
from collections import Counter
import exp028_pretrain_gate as gate
import run_exp028_chain as chain
from coderl_lab.train.exp028_v3_7b_grpo import activated
from coderl_lab.train import exp027_matched_reward as matched
from coderl_lab.train import exp028_cpu_guard as guard
from coderl_lab.train import exp028_durable_reward as durable
from coderl_lab.analysis.exp028_protocol import once,rows,sha

OLD=Path('/home/you/projects/CodeRL-Lab-track-train017/artifacts/exp028_v2')

def prepare(root):
    if root.name!='exp028_v3' or root.parent!=OLD.parent:raise ValueError('new v3 artifact directory required')
    if root.exists():return
    chain.verify_data(OLD)
    summary=json.loads((OLD/'pretrain_generation/summary.json').read_text())
    if summary['checkpoint_sha256']!=sha(OLD/'pretrain_generation/checkpoint.jsonl'):raise ValueError('frozen SFT generation changed')
    stage=root.with_name(root.name+'.preparing')
    if stage.exists():raise FileExistsError('partial preparation retained for audit')
    stage.mkdir()
    for directory in ('data','external96','pretrain_generation'):
        shutil.copytree(OLD/directory,stage/directory)
    once(stage/'generation_reuse.json',{'source':str(OLD/'pretrain_generation'),
          'checkpoint_sha256':summary['checkpoint_sha256'],'reused_gate_generations':288,
          'new_gate_generations':0,'v2_gate_verdicts_reused':False,'v2_models_reused':False})
    stage.rename(root)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('gate','chain'),default='chain')
    p.add_argument('--root',type=Path,required=True);p.add_argument('--publish-github',action='store_true');a=p.parse_args()
    if os.uname().nodename!='Tang':raise RuntimeError('Tang only')
    prepare(a.root)
    if (a.root/'chain_failure.json').exists() or (a.root/'gate_failure.json').exists():raise RuntimeError('v3 stopped; no automatic retries')
    work=Path(__file__).resolve().parents[1]
    argv=sys.argv[:];old_child=chain.child;old_render=chain.render;old_protocol=chain.protocol
    old_verify=chain.verify_model;old_evaluate=chain.evaluate;old_report=chain.REPORT
    def child(cmd,*args,**kwargs):
        cmd=list(cmd)
        if '-m' in cmd:
            i=cmd.index('-m')+1
            if cmd[i]=='coderl_lab.train.exp028_7b_grpo':cmd[i]='coderl_lab.train.exp028_v3_7b_grpo'
        elif len(cmd)>1 and Path(cmd[1]).name=='exp028_pretrain_gate.py':
            cmd=[cmd[0],str(Path(__file__).resolve()),'--phase','gate',*cmd[2:]]
        return old_child(cmd,*args,**kwargs)
    def verify(root,arm,seed):
        result=old_verify(root,arm,seed);d=root/f'{arm}_seed{seed}'
        s=json.loads((d/'summary.json').read_text())
        if s['execution_protocol']!=matched.protocol() or s.get('resource_protocol_amendment')!='EXP028-CPU15-memory256-wall60-v3':
            raise ValueError('mixed training execution protocol')
        if len(list((d/'pending_groups').glob('group_*.json')))!=32 or len(list((d/'reward_groups_ready').glob('group_*.json')))!=32:
            raise ValueError('missing durable group evidence')
        rr=rows(d/'reward_attempts.jsonl')
        result['resource_failures']=dict(Counter(r['verdict'] for r in rr if r['verdict'] in ('cpu_budget_exceeded','memory_budget_exceeded')))
        result['v3_guard_sha256']=sha(guard.__file__);result['v3_journal_sha256']=sha(durable.__file__)
        return result
    def evaluate(*args,**kwargs):
        r=old_evaluate(*args,**kwargs)
        r['training_resource_protocol']=matched.protocol()
        r['historical_v2_results_used_as_control']=False
        return r
    def render(final):
        prefix=['# v3资源协议修订','',
          '两臂统一15秒CPU软预算/16秒硬预算，256MiB内存、1CPU、60秒宿主墙钟保护。明确CPU/OOM预算耗尽作为资源失败；宿主超时或未知退出仍中止。',
          '这是与v2不同的公开协议，v2训练和分数不作为本轮对照。复用冻结SFT检查生成，不复用历史判分；重新检查开训门槛，四次训练均从同一SFT开始。',
          '每组候选执行前已在Tang持久化。原始候选和训练测试不上传。资源失败不等同已证明的算法语义错误。','']
        for r in final.get('training',[]):prefix.append(f"{r['arm']} seed{r['seed']}：资源失败 {r.get('resource_failures',{})}；完整候选组32。")
        return '\n'.join(prefix)+'\n\n'+old_render(final)
    with activated():
        try:
            amendment={'id':'EXP028-CPU15-memory256-wall60-v3','protocol':matched.protocol(),
              'entry_sha256':sha(__file__),'training_entry_sha256':sha(work/'src/coderl_lab/train/exp028_v3_7b_grpo.py'),
              'data_changed':False,'thresholds_changed':False,'CPU_budget_newly_specified':True,
              'memory_or_host_wall_limits_increased':False,'four_trainings_from_SFT':True}
            once(a.root/'resource_amendment.json',amendment)
            if a.phase=='gate':
                try:
                    gate.main_run(a.root,work)
                    counts=Counter()
                    for r in rows(a.root/'pretrain_scored.jsonl'):
                        if r['probe_key']:
                            obs=json.loads((a.root/'training_observations'/(r['probe_key']+'.json')).read_text())
                            error=obs['probe'].get('load_error')
                            if error in ('cpu_budget_exceeded','memory_budget_exceeded'):counts[error]+=1
                    once(a.root/'gate_resource_failures.json',dict(counts))
                except Exception as e:
                    once(a.root/'gate_failure.json',{'error':type(e).__name__,'reason':str(e),'training_started':False});raise
            else:
                chain.child=child;chain.render=render;chain.protocol=matched.protocol
                chain.verify_model=verify;chain.evaluate=evaluate
                chain.REPORT='docs/experiment-logs/2026-10-18-exp028-v3-coverage-result.md'
                sys.argv=[argv[0],'--root',str(a.root)]+(['--publish-github'] if a.publish_github else [])
                chain.main()
        finally:
            chain.child=old_child;chain.render=old_render;chain.protocol=old_protocol
            chain.verify_model=old_verify;chain.evaluate=old_evaluate;chain.REPORT=old_report;sys.argv=argv
if __name__=='__main__':main()
