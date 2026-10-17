"""Run the frozen EXP-024 training loop with only dependency classification fixed.

Supplementary active-source identity fields disambiguate the legacy trainer's
reward_source_sha256 field (which refers to a helper rather than active reward).
The wrapper restores imported globals in finally, so imports have no side effects.
"""
from pathlib import Path
import hashlib
from coderl_lab.train import exp024_7b_grpo as baseline
from coderl_lab.train import exp026_reward as corrected

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    saved_reward=baseline.reward_factory;saved_preflight=baseline.preflight
    def preflight(*args,**kwargs):
        out=saved_preflight(*args,**kwargs)
        out.update({'experiment':'EXP-026-dependency-family-fix',
                    'base_training_protocol':'TRAIN-017 / EXP-024',
                    'active_reward_source_sha256':digest(corrected.__file__),
                    'training_wrapper_sha256':digest(__file__),
                    'active_training_loop_sha256':digest(baseline.__file__),
                    'legacy_reward_source_field_is_active':False})
        return out
    baseline.reward_factory=corrected.reward_factory;baseline.preflight=preflight
    try:baseline.main()
    finally:
        baseline.reward_factory=saved_reward;baseline.preflight=saved_preflight

if __name__=='__main__':main()
