"""Do not let 7B SFT solved@4 accidentally be copied from GRPO again."""
import pytest

from coderl_lab.analysis.train016_score_summary_repair import recompute


def synthetic_12_by_4_by_2():
    rows=[]
    for i in range(12):
        for arm in ("sft","grpo"):
            for sample in range(4):
                # SFT can never solve task0, GRPO can; every other task has
                # a correct sample at sample=1 for both model arms.
                passed = (sample==1 and (arm=="grpo" or i!=0))
                rows.append({
                    "task_id":f"heldout/test/{i}","arm":arm,
                    "sample_id":sample,"all_passed":passed,
                    "partial_reward":1. if passed else .1,
                })
    return rows


def test_recompute_both_arms_from_the_same_immutable_96_verdicts():
    r=recompute(synthetic_12_by_4_by_2())
    assert r["tasks"]==12 and r["candidates"]==96
    assert r["scores_by_arm"]["sft"]["solved_at4"]==11
    assert r["scores_by_arm"]["grpo"]["solved_at4"]==12
    assert r["paired_task_bootstrap"]["pass_at4_diff"]["estimate"]==pytest.approx(1/12)


def test_fail_closed_if_scored_arms_or_samples_missing():
    rows=synthetic_12_by_4_by_2()
    with pytest.raises(ValueError,match="96"):
        recompute(rows[:-1])
    rows[0]["arm"]="base"
    with pytest.raises(ValueError,match="bad/duplicate"):
        recompute(rows)
