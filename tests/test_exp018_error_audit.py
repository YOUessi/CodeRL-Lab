import pytest

from coderl_lab.analysis.exp018_train017_failure_audit import aggregate
from coderl_lab.analysis.train017_7b_generate import ARMS,SEEDS,N_DRAWS


def make_matrix():
    frozen={}
    scores={}
    for i in range(24):
        q=f"heldout/group/{i}"
        for seed in SEEDS:
            for sample in range(N_DRAWS):
                for arm in ARMS:
                    key=(q,seed,sample,arm)
                    passed=(arm=="new_grpo" and sample==0) or (arm=="sft" and sample==1 and i<10)
                    output="def solve(x): return 1" if passed else "def solve(x): return 0"
                    frozen[key]={
                        "answer":output,"answer_sha256":f"sha_{i}_{seed}_{sample}_{arm}",
                        "generated_tokens":30,
                    }
                    scores[key]={
                        "answer_sha256":frozen[key]["answer_sha256"],
                        "all_passed":passed,"reward":1. if passed else .1,
                        "python_valid":True,"entry_valid":True,"verdict":"none" if passed else "wrong_answer",
                    }
    return frozen,scores


def test_24_task_2_seed_three_arm_margins():
    f,s=make_matrix()
    result=aggregate(f,s)
    assert result["all_candidates"]==576
    assert result["by_arm"]["new_grpo"]["by_family"]["group"]["correct"]==48
    assert result["by_arm"]["sft"]["by_family"]["group"]["correct"]==20
    comparison=result["coverage_transitions"]["new_grpo_minus_sft"]
    assert comparison["extra_correct_candidates"]==28
    assert comparison["task_seed_pair_solved4_transitions"]=={
        "0_to_1":28,"1_to_1":20,
    }


def test_missing_or_mutated_scores_rejected():
    generated,scored=make_matrix()
    del scored[next(iter(scored))]
    with pytest.raises(ValueError,match="576"):
        aggregate(generated,scored)
    frozen,scored=make_matrix()
    del frozen[next(iter(frozen))]
    with pytest.raises(ValueError,match="576"):
        aggregate(frozen,scored)
