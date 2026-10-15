"""Frozen 576-code posthoc GRPO coverage audit quality gates (CPU-only)."""
from copy import deepcopy
import hashlib

import pytest

from coderl_lab.analysis.exp018_frozen_failure_audit import (
    MODEL_ARMS,SEEDS,N_TASKS,SAMPLES_PER_TASK,
    bootstrap,compute,exact_sign_test,row_index,
)


def toy_24_tasks():
    generated=[]
    scored=[]
    ids=[f"heldout/toy_algorithm/{i}" for i in range(N_TASKS)]
    for i,q in enumerate(ids):
        for seed in SEEDS:
            for j in range(SAMPLES_PER_TASK):
                for arm in MODEL_ARMS:
                    # 12 old covered tasks, new more reliable within them,
                    # plus one additional heldout task solved by new model.
                    passed=(i<12 and (j==0 or arm=="new_grpo" and j==1))
                    if arm=="new_grpo" and i==12 and j==0:
                        passed=True
                    raw=f"def solve(x):\n    return {i+j}\n# {seed} {arm}"
                    h=hashlib.sha256(raw.encode()).hexdigest()
                    key={"task_id":q,"seed":seed,"sample_id":j,"arm":arm}
                    generated.append({
                        **key,"answer":raw,"answer_sha256":h,
                        "generated_tokens":24,
                    })
                    scored.append({
                        **key,"answer_sha256":h,"all_passed":bool(passed),
                        "python_valid":True,"entry_valid":True,
                        "reward":1. if passed else .1,
                        "verdict":"none" if passed else "wrong_answer",
                        "normalized_code_sha256":h,
                    })
    return ids,generated,scored


def test_true_pairing_and_reliability_vs_coverage_decomposition():
    ids,g,s=toy_24_tasks()
    outcome=compute(g,s,ids)
    assert outcome["task_count"]==24
    assert outcome["candidate_count"]==576
    assert outcome["arm_metrics"]["sft"]["candidate_correct"]==24
    assert outcome["arm_metrics"]["new_grpo"]["candidate_correct"]==50
    assert outcome["arm_metrics"]["sft"]["solved_at_k"][3]==24
    assert outcome["arm_metrics"]["new_grpo"]["solved_at_k"][3]==26
    change=outcome["paired_comparison"]["new_grpo_minus_sft"]
    assert change["sample_correctness_net_gain"]==26
    assert change["task_seed_newly_covered"]==2
    assert change["task_seed_lost_coverage"]==0
    assert change["already_covered_reliability_gain"]==24
    assert change["task_cluster_bootstrap"]["solved_at4"]["observed"]==pytest.approx(1/24)
    assert outcome["raw_generated_code_exported"] is False


def test_reject_duplicate_wrong_code_hash_or_missing_model():
    ids,g,s=toy_24_tasks()
    with pytest.raises(ValueError,match="576"):
        compute(g[:-1],s,ids)
    altered=deepcopy(g)
    altered[0]["answer"]="changed_after_freezing"
    with pytest.raises(ValueError,match="mutated"):
        compute(altered,s,ids)
    dup=deepcopy(g)
    dup[0]=deepcopy(dup[1])
    with pytest.raises(ValueError,match="duplicated"):
        compute(dup,s,ids)


def test_task_cluster_ci_and_exact_sign_edges():
    q=bootstrap([0.]*24,reps=100)
    assert q["observed"]==q["ci95_low"]==q["ci95_high"]==0
    assert q["cluster"]=="one_task_with_two_correlated_seeds"
    with pytest.raises(ValueError,match="24"):
        bootstrap([0.]*48)
    assert exact_sign_test(0,0)==1
    assert exact_sign_test(5,0)==pytest.approx(.0625)
