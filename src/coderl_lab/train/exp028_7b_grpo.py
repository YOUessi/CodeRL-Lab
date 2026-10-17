"""EXP028 swaps only the verified training distribution, not GRPO/reward code."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from coderl_lab.train import exp024_7b_grpo as baseline
from coderl_lab.train import exp027_matched_reward as matched
from coderl_lab.datasets.exp028_coverage import EXPERIMENT
from coderl_lab.analysis.exp028_protocol import ARMS,SEEDS,sha


def validate_controls(cfg,arm):
    if arm not in ARMS or cfg['experiment'].get('coverage_arm')!=arm:raise ValueError('coverage mismatch')
    if cfg['experiment']['seed'] not in SEEDS or cfg['experiment']['reward_arm']!='strict':raise ValueError('seed/reward mismatch')
    fixed={'num_generations':4,'per_device_train_batch_size':4,'gradient_accumulation_steps':1,'max_completion_length':256,
           'learning_rate':1e-5,'temperature':.8,'top_p':.95,'beta':0.,'loss_type':'grpo'}
    if any(cfg['grpo'].get(k)!=v for k,v in fixed.items()):raise ValueError('training budget/recipe changed')
    if cfg['runs']['formal_steps']!=32 or cfg['reward'].get('wall_guard_seconds')!=60.:raise ValueError('steps/execution protocol changed')


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--coverage',choices=ARMS,required=True)
    p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--gate-report',type=Path,required=True)
    opts,rest=p.parse_known_args()
    gate=json.loads(opts.gate_report.read_text())
    if gate.get('passed') is not True or gate.get('status')!='gate_passed':raise ValueError('pretraining gate not passed')
    original_argv=sys.argv[:];old_preflight=baseline.preflight;old_factory=baseline.reward_factory
    old_experiment=baseline.FROZEN_DATA_EXPERIMENT
    baseline.FROZEN_DATA_EXPERIMENT=EXPERIMENT
    def preflight(cfg,root,config_path):
        validate_controls(cfg,opts.coverage)
        data=Path(root).parent
        if sha(data/'manifest.json')!=gate['data_manifest_sha256']:raise ValueError('data differ from gated data')
        manifest=json.loads((data/'manifest.json').read_text())
        for rel,h in manifest['files'].items():
            if sha(data/rel)!=h:raise ValueError('frozen data drift')
        result=old_preflight(cfg,root,config_path)
        result.update({'experiment':'EXP028-coverage-training','coverage_arm':opts.coverage,'training_seed':cfg['experiment']['seed'],
                       'reward_variant':'corrected','execution_protocol':matched.protocol(),
                       'active_reward_source_sha256':sha(matched.__file__),'active_training_loop_sha256':sha(baseline.__file__),
                       'training_wrapper_sha256':sha(__file__),'gate_report_sha256':sha(opts.gate_report),
                       'legacy_reward_source_field_is_active':False})
        return result
    baseline.preflight=preflight
    baseline.reward_factory=lambda logfile:matched.reward_factory(logfile,'corrected',opts.cache_root)
    sys.argv=[original_argv[0],*rest]
    try:baseline.main()
    finally:
        baseline.preflight=old_preflight;baseline.reward_factory=old_factory
        baseline.FROZEN_DATA_EXPERIMENT=old_experiment;sys.argv=original_argv
if __name__=='__main__':main()
