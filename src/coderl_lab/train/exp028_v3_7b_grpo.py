"""EXP028 v3: same data/rewards/GRPO, explicit CPU resource budget and journaling."""
from contextlib import contextmanager
import hashlib
from coderl_lab.train import exp028_7b_grpo as training
from coderl_lab.train import exp027_matched_reward as matched
from coderl_lab.train import exp028_cpu_guard as guard
from coderl_lab.train import exp028_durable_reward as durable
from coderl_lab.analysis.exp028_protocol import sha

@contextmanager
def activated():
    previous_guard=matched.guarded;previous_factory=matched.reward_factory;previous_protocol=matched.protocol
    previous_preflight=training.baseline.preflight
    def protocol():
        p=previous_protocol()
        p.update({'id':'EXP028-CPU15-memory256-wall60-v3','cpu_seconds':guard.CPU_SECONDS,
                  'cpu_hard_seconds':guard.CPU_HARD_SECONDS,'memory_mib':256,
                  'driver_sha256':hashlib.sha256(guard.DRIVER.encode()).hexdigest(),
                  'guard_sha256':sha(guard.__file__),'evidence_source_sha256':sha(durable.__file__),
                  'cpu_budget_exit':'verified_resource_failure','host_wall_expiry':'indeterminate_abort_no_reward'})
        return p
    def preflight(*args,**kwargs):
        p=previous_preflight(*args,**kwargs)
        p.update({'resource_protocol_amendment':'EXP028-CPU15-memory256-wall60-v3',
                  'resource_wrapper_sha256':sha(__file__),'resource_guard_sha256':sha(guard.__file__),
                  'evidence_wrapper_sha256':sha(durable.__file__),'historical_v2_scores_not_reused':True})
        return p
    matched.guarded=guard;matched.reward_factory=durable.reward_factory;matched.protocol=protocol
    training.baseline.preflight=preflight
    try:yield
    finally:
        matched.guarded=previous_guard;matched.reward_factory=previous_factory;matched.protocol=previous_protocol
        training.baseline.preflight=previous_preflight

def main():
    with activated():training.main()
if __name__=='__main__':main()
