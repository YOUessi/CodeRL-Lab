"""Same frozen 7B training loop; only the dependency classifier differs by arm."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from coderl_lab.train import exp024_7b_grpo as baseline
from coderl_lab.train import exp027_matched_reward as matched


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--variant',choices=matched.VARIANTS,required=True)
    p.add_argument('--cache-root',type=Path,required=True)
    opts,rest=p.parse_known_args()
    keep_argv=sys.argv[:];keep_factory=baseline.reward_factory;keep_preflight=baseline.preflight
    def checked_preflight(cfg,root,config_path):
        if cfg['experiment'].get('matched_variant')!=opts.variant:raise ValueError('variant/config mismatch')
        if cfg['reward'].get('wall_guard_seconds')!=matched.WALL_SECONDS:raise ValueError('wall protocol mismatch')
        if cfg['experiment']['seed'] not in (43,44):raise ValueError('unregistered training seed')
        result=keep_preflight(cfg,root,config_path)
        result.update({'experiment':'EXP027-matched-execution','matched_variant':opts.variant,
                       'training_seed':cfg['experiment']['seed'],'execution_protocol':matched.protocol(),
                       'execution_cache_root':str(opts.cache_root.resolve()),
                       'active_reward_source_sha256':matched.file_sha(matched.__file__),
                       'training_wrapper_sha256':matched.file_sha(__file__),
                       'active_training_loop_sha256':matched.file_sha(baseline.__file__),
                       'legacy_reward_source_field_is_active':False})
        return result
    baseline.preflight=checked_preflight
    baseline.reward_factory=lambda logfile:matched.reward_factory(logfile,opts.variant,opts.cache_root)
    sys.argv=[keep_argv[0],*rest]
    try:baseline.main()
    finally:
        baseline.preflight=keep_preflight;baseline.reward_factory=keep_factory;sys.argv=keep_argv

if __name__=='__main__':main()
