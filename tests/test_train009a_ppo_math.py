from __future__ import annotations

import math
import pytest

from coderl_lab.train.ppo_math import (
    terminal_kl_rewards,
    generalized_advantages,
    ppo_clip_surrogate,
    normalize_advantages,
)


def test_kl_reward_terminal_only_and_reference_identity():
    values=terminal_kl_rewards(
        old_logprobs=[-1,-2,-1],reference_logprobs=[-1,-2,-1],
        scalar_reward=0.7,kl_coefficient=0.1,reward_scale=1,reward_cap=2,
    )
    assert values==pytest.approx([0,0,0.7])
    penalty=terminal_kl_rewards(
        old_logprobs=[-0.5,-1],reference_logprobs=[-1,-1.5],
        scalar_reward=10,kl_coefficient=0.2,reward_scale=1,reward_cap=2,
    )
    assert penalty==pytest.approx([-0.1,1.9])
    with pytest.raises(ValueError):
        terminal_kl_rewards(
            old_logprobs=[1],reference_logprobs=[],
            scalar_reward=0,kl_coefficient=0.1,reward_scale=1,reward_cap=1,
        )


def test_gae_propagates_terminal_reward_through_tokens_and_value_baseline():
    adv,ret=generalized_advantages([0,0,1],[0,0,0],gamma=1,gae_lambda=1)
    assert adv==ret==pytest.approx([1,1,1])
    adv,ret=generalized_advantages([0,0,1],[0,0,0],gamma=1,gae_lambda=0.5)
    assert adv==ret==pytest.approx([0.25,0.5,1])
    adv,ret=generalized_advantages([0,1],[0.4,0.6],gamma=1,gae_lambda=1)
    assert adv==pytest.approx([0.6,0.4])
    assert ret==pytest.approx([1,1])
    with pytest.raises(ValueError):
        generalized_advantages([0],[0,0])


def test_ppo_clip_preserves_positive_and_negative_advantage_direction():
    ratio=math.log(2.0)
    positive=ppo_clip_surrogate(
        new_logprob=ratio,old_logprob=0,advantage=1,clip_epsilon=0.2,
    )
    assert positive==pytest.approx(1.2)
    negative=ppo_clip_surrogate(
        new_logprob=ratio,old_logprob=0,advantage=-1,clip_epsilon=0.2,
    )
    assert negative==pytest.approx(-2.0)
    too_low=ppo_clip_surrogate(
        new_logprob=math.log(0.2),old_logprob=0,advantage=-1,clip_epsilon=0.2,
    )
    assert too_low==pytest.approx(-0.8)
    with pytest.raises(ValueError):
        ppo_clip_surrogate(
            new_logprob=40,old_logprob=0,advantage=1,clip_epsilon=0.2,
        )


def test_normalized_advantages_center_and_constant_returns_zero():
    data=normalize_advantages([-3,-1,2,8])
    assert abs(sum(data)) < 1e-6
    assert sum(x*x for x in data)/len(data)==pytest.approx(1,abs=1e-6)
    assert normalize_advantages([5,5])==[0,0]
    with pytest.raises(ValueError):
        normalize_advantages([])


def test_tensor_ppo_surrogate_gradient_matches_cpu_reference_if_torch_available():
    torch=pytest.importorskip("torch")
    advantage=torch.tensor([1.0,-1.0],requires_grad=False)
    old=torch.tensor([-0.1,-0.1])
    new=torch.tensor([math.log(2)-0.1,math.log(0.2)-0.1],requires_grad=True)
    ratio=(new-old).exp()
    objective=torch.minimum(ratio*advantage,ratio.clamp(0.8,1.2)*advantage)
    assert objective.detach().tolist()==pytest.approx([1.2,-0.8])
    loss=-objective.mean()
    loss.backward()
    assert new.grad.abs().max().item()==pytest.approx(0)
