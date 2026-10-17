"""Explicit v2 execution, never overwriting the original stopped EXP028 run.
This changes no data, model, sampling, reward weights, thresholds, or budgets.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import exp028_pretrain_gate as gate
import run_exp028_chain as chain
from coderl_lab.train import exp027_matched_reward as matched
from coderl_lab.train import exp028_resource_guard as resource
from coderl_lab.analysis.exp028_protocol import once,rows,sha,canonical


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('gate','chain'),default='chain');p.add_argument('--root',type=Path,required=True);p.add_argument('--publish-github',action='store_true');a=p.parse_args()
    if a.root.name!='exp028_v2':raise ValueError('resource amendment must not overwrite original run')
    work=Path(__file__).resolve().parents[1]
    before=matched.guarded;matched.guarded=resource
    old_child=chain.child;old_render=chain.render;argv=sys.argv[:]
    def amended_child(cmd,*args,**kwargs):
        cmd=list(cmd)
        if '-m' in cmd:
            i=cmd.index('-m')+1
            if cmd[i]=='coderl_lab.train.exp028_7b_grpo':cmd[i]='coderl_lab.train.exp028_v2_7b_grpo'
        elif len(cmd)>1 and Path(cmd[1]).name=='exp028_pretrain_gate.py':
            cmd=[cmd[0],str(Path(__file__).resolve()),'--phase','gate',*cmd[2:]]
        return old_child(cmd,*args,**kwargs)
    def render(final):
        return '# 资源协议修订记录\n\n本次使用确认OOM分类v2，两臂内存256MiB、1CPU、60秒墙钟安全界限相同。退出137只有获得Docker OOMKilled证据时才作为资源预算失败记奖励0；其它不明确退出仍停止。原EXP028停止记录保留。SFT检查的288份冻结生成整体复用，没有额外重采样；参考解和候选重新评分。\n\n'+old_render(final)
    try:
        once(a.root/'resource_amendment.json',{'id':'EXP028-confirmed-OOM-v2','guard_sha256':sha(resource.__file__),
                                             'entry_sha256':sha(__file__),'training_entry_sha256':sha(work/'src/coderl_lab/train/exp028_v2_7b_grpo.py'),
                                             'thresholds_changed':False,'data_changed':False,'budget_changed':False,'fresh_gate_generations':0,'reused_gate_generations':288})
        if a.phase=='gate':
            try:
                result=gate.main_run(a.root,work)
                limited=[]
                for r in rows(a.root/'pretrain_scored.jsonl'):
                    if not r['probe_key']:continue
                    observation=json.loads((a.root/'training_observations'/(r['probe_key']+'.json')).read_text())
                    if observation['probe'].get('load_error')=='memory_budget_exceeded':limited.append({'task_id':r['task_id'],'sample_id':r['sample_id']})
                once(a.root/'gate_resource_failures.json',{'memory_budget_failures':limited,'count':len(limited),'semantic_error_claimed':False})
            except Exception as e:
                once(a.root/'gate_failure.json',{'error':type(e).__name__,'reason':str(e),'training_started':False});raise
        else:
            chain.child=amended_child;chain.render=render
            sys.argv=[argv[0],'--root',str(a.root)]+(['--publish-github'] if a.publish_github else [])
            chain.main()
    finally:
        matched.guarded=before;chain.child=old_child;chain.render=old_render;sys.argv=argv
if __name__=='__main__':main()
