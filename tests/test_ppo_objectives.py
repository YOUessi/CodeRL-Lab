from __future__ import annotations

import math

import numpy as np
import pytest

from coderl_lab.train.ppo_objectives import (
    assign_terminal_reward,
    frozen_reference_kl,
    generalized_advantage_estimate,
    ppo_clipped_policy_loss,
    ppo_clipped_value_loss,
    ppo_combined_objective,
    validate_termination,
)


def test_terminal_audit_rejects_masked_all_truncated_rollouts():
    mask=np.array([[1,1,1],[1,1,0]],dtype=int)
    term=np.zeros_like(mask)
    with pytest.raises(ValueError,match="all rollouts truncated"):
        validate_termination(valid_mask=mask,terminal_mask=term)
    report=validate_termination(valid_mask=mask,terminal_mask=term,
                                require_any_completed=False)
    assert report["truncated_rollouts"]==2
    assert report["all_truncated"] is True


def test_terminal_reward_not_naively_paid_to_unfinished_model_output():
    valid=np.array([[1,1,1],[1,1,0]],dtype=int)
    terminal=np.array([[0,0,1],[0,0,0]],dtype=int)
    rewards,report=assign_terminal_reward(
        learned_reward=np.array([0.9,0.95]),
        valid_mask=valid,terminal_mask=terminal,truncated_penalty=-1.0,
    )
    np.testing.assert_allclose(rewards,[[0,0,0.9],[0,-1,0]])
    assert report["terminated_fraction"]==pytest.approx(0.5)
    assert report["truncation_reward_policy"]=="fixed_penalty_not_learned_rm"


def test_terminal_cannot_be_outside_valid_or_followed_by_live_tokens():
    mask=[[1,1,0]]
    with pytest.raises(ValueError,match="padded"):
        validate_termination(valid_mask=mask,terminal_mask=[[0,0,1]])
    with pytest.raises(ValueError,match="last valid"):
        validate_termination(valid_mask=[[1,1,1]],terminal_mask=[[0,1,0]])
    with pytest.raises(ValueError,match="contiguous"):
        validate_termination(valid_mask=[[1,0,1]],terminal_mask=[[0,0,1]])
    with pytest.raises(ValueError,match="empty"):
        validate_termination(valid_mask=[[0,0]],terminal_mask=[[0,0]])


def test_masked_gae_natural_eos_bypasses_bootstrap_and_truncation_bootstraps():
    value=np.array([[0.2,0.3,0.4]])
    reward=np.array([[0,1.]])
    eos=np.array([[0,1]])
    mask=np.array([[1,1]])
    adv,returns=generalized_advantage_estimate(
        rewards=reward,values=value,valid_mask=mask,terminal_mask=eos,
        gamma=1,gae_lambda=1,
    )
    np.testing.assert_allclose(adv,[[0.8,0.7]])
    np.testing.assert_allclose(returns,[[1,1]])
    incomplete,ret=generalized_advantage_estimate(
        rewards=reward,values=value,valid_mask=mask,
        terminal_mask=np.array([[0,0]]),
        gamma=1,gae_lambda=1,
    )
    np.testing.assert_allclose(incomplete,[[1.2,1.1]])
    np.testing.assert_allclose(ret,[[1.4,1.4]])


def test_gae_resets_after_padding_and_ignores_arbitrary_padded_values():
    r=np.array([[0,1,999]],dtype=float)
    v=np.array([[0.0,0.0,0.0,777]])
    adv,returns=generalized_advantage_estimate(
        rewards=r,values=v,valid_mask=[[1,1,0]],terminal_mask=[[0,1,0]],
        gamma=1,gae_lambda=1,
    )
    np.testing.assert_allclose(adv,[[1,1,0]])
    np.testing.assert_allclose(returns,[[1,1,0]])


def test_clipped_ppo_policy_limits_positive_and_negative_advantage():
    ratio=np.array([[1.5,0.5]])
    p=ppo_clipped_policy_loss(
        new_logprobs=np.log(ratio),old_logprobs=np.zeros((1,2)),
        advantages=[[1,-1]],valid_mask=[[1,1]],clip_epsilon=.2,
    )
    assert p["policy_loss"]==pytest.approx(-0.2)
    assert p["clip_fraction"]==pytest.approx(1)
    assert p["nonzero_advantage_fraction"]==pytest.approx(1)
    assert p["sampled_approx_kl"]>=0


def test_clipped_value_loss_uses_conservative_worse_of_two_errors():
    v=ppo_clipped_value_loss(
        new_values=[[1]],old_values=[[0]],returns=[[1]],
        valid_mask=[[1]],clip_epsilon=.2,
    )
    assert v["value_loss"]==pytest.approx(0.32)
    assert v["value_clip_fraction"]==pytest.approx(1)


def test_explicit_frozen_reference_kl_not_same_as_unregularized_actor():
    assert frozen_reference_kl(new_logprobs=[[0.]],ref_logprobs=[[0.]],
                               valid_mask=[[1]])==pytest.approx(0)
    kl=frozen_reference_kl(
        new_logprobs=[[math.log(.5)]],ref_logprobs=[[0]],
        valid_mask=[[1]],
    )
    assert kl==pytest.approx(1-math.log(2))
    policy={"policy_loss":.5}
    value={"value_loss":.4}
    combined=ppo_combined_objective(
        policy=policy,value=value,frozen_ref_kl=kl,entropy_mean=1.,
        value_coefficient=.5,kl_coefficient=.02,entropy_coefficient=.01)
    assert combined["total_loss"]==pytest.approx(.5+.5*.4+.02*kl-.01)


def test_zero_effective_tokens_and_nonfinite_values_block_false_ppo_updates():
    with pytest.raises(ValueError,match="zero effective completion tokens"):
        ppo_clipped_policy_loss(
            new_logprobs=[[0]],old_logprobs=[[0]],advantages=[[1]],
            valid_mask=[[0]])
    with pytest.raises(ValueError,match="finite"):
        ppo_clipped_policy_loss(
            new_logprobs=[[float("nan")]],old_logprobs=[[0]],
            advantages=[[1]],valid_mask=[[1]])
    with pytest.raises(ValueError,match="importance ratio"):
        ppo_clipped_policy_loss(
            new_logprobs=[[99]],old_logprobs=[[0]],
            advantages=[[1]],valid_mask=[[1]])
    with pytest.raises(ValueError,match="bootstrap"):
        generalized_advantage_estimate(
            rewards=[[1]],values=[[0]],valid_mask=[[1]],
            terminal_mask=[[1]])
    with pytest.raises(ValueError,match="non-negative"):
        ppo_combined_objective(
            policy={"policy_loss":0},value={"value_loss":0},
            frozen_ref_kl=0,entropy_mean=0,kl_coefficient=-1)
