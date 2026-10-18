"""One-shot diagnostic replay of the failed EXP028 v2 first generation group.

No optimizer update, no candidate execution, no benchmark tests. This reconstructs
candidates using the frozen trainer/config; missing historical candidate hashes
mean byte identity with the lost original group cannot be proved. Never replaces
historical failure records. Run only on Tang with the original v2 source on path.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path
import yaml

class CapturedFirstGroup(BaseException):
    pass

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',required=True,type=Path)
    p.add_argument('--old-root',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    if os.uname().nodename!='Tang':raise RuntimeError('Tang only')
    if a.out.exists():raise FileExistsError('diagnostic cannot overwrite any run')
    old=json.loads((a.old_root/'active_code.json').read_text())
    for rel,h in old['files'].items():
        if sha(a.source_root/rel)!=h:raise ValueError('original source changed: '+rel)
    cfg=yaml.safe_load((a.old_root/'broad_seed43_config.yaml').read_text())
    if cfg['experiment']['seed']!=43 or cfg['experiment']['coverage_arm']!='broad':raise ValueError('wrong replay')
    a.out.mkdir(parents=True)
    cfg['runs']['formal_output']=str(a.out/'diagnostic_run')
    config=a.out/'config.yaml';config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    sys.path.insert(0,str(a.source_root/'src'))
    from coderl_lab.train import exp028_v2_7b_grpo as entry
    from coderl_lab.train import exp027_matched_reward as matched
    original=matched.reward_factory
    def factory(logfile,variant,cache_root):
        def callback(prompts,completions,completion_ids=None,task_id=None,test_cases=None,**kwargs):
            if not(len(prompts)==len(completions)==len(task_id)==len(test_cases)==4):raise ValueError('expected first complete group')
            result={'status':'captured_before_reward_and_optimizer','source_files':old['files'],
                    'variant':variant,'task_ids':task_id,'completions':completions,
                    'test_cases':test_cases,'prompts':prompts,
                    'historical_candidate_identity':'unknown_original_not_persisted',
                    'optimizer_updates':0}
            target=a.out/'first_group.json'
            with target.open('x') as f:
                f.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
            raise CapturedFirstGroup()
        callback.__name__='exp027_corrected_matched_reward'
        return callback
    matched.reward_factory=factory
    saved=sys.argv[:]
    sys.argv=[__file__,'--coverage','broad','--cache-root',str(a.out/'unused_cache'),
              '--gate-report',str(a.old_root/'gate_report.json'),'--arm','strict',
              '--config',str(config),'--data-dir',str(a.old_root/'data/broad'),
              '--sft-adapter','/home/you/projects/CodeRL-Lab-track-train011/artifacts/train011/qlora-ultrachat',
              '--output-dir',str(a.out/'diagnostic_run'),'--max-steps','32','--run-kind','formal']
    try:entry.main()
    except CapturedFirstGroup:print('FIRST_GROUP_CAPTURED_NO_UPDATE',sha(a.out/'first_group.json'),flush=True)
    else:raise RuntimeError('diagnostic did not reach callback')
    finally:matched.reward_factory=original;sys.argv=saved
if __name__=='__main__':main()
