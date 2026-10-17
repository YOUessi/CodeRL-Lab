"""Apply the documented pretraining resource amendment to both EXP028 arms."""
from coderl_lab.train import exp028_7b_grpo as training
from coderl_lab.train import exp027_matched_reward as matched
from coderl_lab.train import exp028_resource_guard as guard
from coderl_lab.analysis.exp028_protocol import sha


def main():
    old_guard=matched.guarded;old_preflight=training.baseline.preflight
    def preflight(*args,**kwargs):
        r=old_preflight(*args,**kwargs)
        r.update({'resource_protocol_amendment':'EXP028-confirmed-OOM-v2','resource_wrapper_sha256':sha(__file__)})
        return r
    matched.guarded=guard;training.baseline.preflight=preflight
    try:training.main()
    finally:matched.guarded=old_guard;training.baseline.preflight=old_preflight
if __name__=='__main__':main()
