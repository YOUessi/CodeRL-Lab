"""Smoke-only natural-EOS guarded GRPO correctness tests, no CUDA required."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from coderl_lab.train.learned_reward_grpo import (
    BoundedLearnedReward,validate_grpo_contract,
)


def config():
    return yaml.safe_load(
        Path("configs/train009c_eos_guarded_grpo_smoke.yaml").read_text())


def test_pre_registered_512_budget_keeps_terminated_only_policy_mask():
    c=config()
    assert c["experiment"]["id"]=="TRAIN-009C"
    a=validate_grpo_contract(c,smoke=True)
    assert a["optimizer_steps"]==2
    assert a["natural_eos_required"] is True
    assert a["gradient_smoke_allow_truncated"] is False
    assert c["generation"]["max_completion_length"]==512
    assert c["training"]["mask_truncated_completions"] is True
    assert c["training"]["beta"]==0.02
    assert c["reward_model"]["require_natural_eos"] is True
    assert c["reward_model"]["eos_token_id"]==151643
    with pytest.raises(ValueError,match="frozen positive max_steps"):
        validate_grpo_contract(c,smoke=False)


def test_only_natural_endoftext_gets_learned_scalar_reward():
    called=[]
    def raw_score(p,c):
        called.append((p,c))
        return 1.0
    scorer=BoundedLearnedReward(
        score_one=raw_score,require_natural_eos=True,
        eos_token_id=151643,empty_reward=-1.0,
    )
    output=scorer(
        prompts=["alpha","beta","gamma","delta"],
        completions=["finished","unfinished","different turn end","finished again"],
        completion_ids=[
            [123,151643,151643],
            [2,3,4],
            [12,151645,14],
            [30,40,151643],
        ],
    )
    assert len(called)==2
    assert output[0]==pytest.approx(output[3])
    assert output[1]==-1 and output[2]==-1
    agg=scorer.aggregate()
    assert agg["natural_eos_terminated"]==2
    assert agg["truncated_completions"]==2
    assert agg["rewarded_rollouts"]==4
    assert "alpha" not in str(scorer.history)
    assert "finished" not in str(scorer.history)


def test_nat_eos_requires_token_level_evidence_no_text_only_shortcuts():
    scorer=BoundedLearnedReward(score_one=lambda p,c:0.5,
        require_natural_eos=True,eos_token_id=151643)
    with pytest.raises(ValueError,match="token IDs"):
        scorer(prompts=["P"],completions=["complete-looking text"])
    with pytest.raises(ValueError,match="does not match"):
        scorer(prompts=["P"],completions=["C"],completion_ids=[[151643],[]])
    with pytest.raises(TypeError,match="row of token IDs"):
        scorer(prompts=["P"],completions=["C"],completion_ids=[151643])


def test_all_truncated_rollouts_are_penalized_not_scored():
    def never_score(*args):
        raise AssertionError("RM must not reward an incomplete response")
    scorer=BoundedLearnedReward(
        score_one=never_score,require_natural_eos=True,eos_token_id=151643)
    v=scorer(
        prompts=["P"]*4,
        completions=["raw unended"]*4,
        completion_ids=[[12,16,17] for _ in range(4)],
    )
    assert v==[-1.0]*4
    assert scorer.aggregate()["natural_eos_terminated"]==0
    assert scorer.aggregate()["avg_bounded_reward_std"]==0
