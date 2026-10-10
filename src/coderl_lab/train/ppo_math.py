"""Small, dependency-free reference math for TRAIN-009A on-policy PPO-Clip.

This is actual token-level PPO with an explicit trainable critic and GAE,
not a renaming of GRPO, DPO or offline preference regression.
These pure functions are used by CPU tests and mirror the Torch objective.
"""
from __future__ import annotations

import math
from typing import Sequence


def terminal_kl_rewards(
    *,
    old_logprobs: Sequence[float],
    reference_logprobs: Sequence[float],
    scalar_reward: float,
    kl_coefficient: float,
    reward_scale: float,
    reward_cap: float,
) -> list[float]:
    if len(old_logprobs) != len(reference_logprobs) or not old_logprobs:
        raise ValueError("one reference log-prob is needed for each sampled action")
    if kl_coefficient < 0 or reward_scale <= 0 or reward_cap <= 0:
        raise ValueError("KL and reward coefficients must be nonnegative/positive")
    nums = [*old_logprobs, *reference_logprobs, scalar_reward]
    if not all(math.isfinite(float(x)) for x in nums):
        raise ValueError("non-finite PPO reward inputs")
    shaped = [
        -kl_coefficient * (float(old) - float(ref))
        for old, ref in zip(old_logprobs, reference_logprobs, strict=True)
    ]
    terminal = min(reward_cap, max(-reward_cap, reward_scale * scalar_reward))
    shaped[-1] += terminal
    return shaped


def generalized_advantages(
    rewards: Sequence[float],
    values: Sequence[float],
    *,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> tuple[list[float], list[float]]:
    """On-policy Monte-Carlo episode; terminal V(next)=0, no value leakage."""
    if not rewards or len(rewards) != len(values):
        raise ValueError("one value prediction per sampled token")
    if not (0 <= gamma <= 1 and 0 <= gae_lambda <= 1):
        raise ValueError("invalid GAE discount/lambda")
    if not all(math.isfinite(float(v)) for v in (*rewards, *values)):
        raise ValueError("nonfinite reward/critic values")
    n = len(rewards)
    advantage = [0.0] * n
    carry = 0.0
    for t in range(n - 1, -1, -1):
        next_value = values[t + 1] if t + 1 < n else 0.0
        td_error = rewards[t] + gamma * next_value - values[t]
        carry = td_error + gamma * gae_lambda * carry
        advantage[t] = carry
    returns = [advantage[t] + values[t] for t in range(n)]
    return advantage, returns


def ppo_clip_surrogate(
    *,
    new_logprob: float,
    old_logprob: float,
    advantage: float,
    clip_epsilon: float,
) -> float:
    """Min objective, differentiable Torch counterpart is in ppo_online.py."""
    if not 0 < clip_epsilon < 1:
        raise ValueError("PPO ratio clip must be between zero and one")
    if not all(math.isfinite(x) for x in (new_logprob, old_logprob, advantage)):
        raise ValueError("PPO values must be finite")
    diff = new_logprob - old_logprob
    if abs(diff) > 20:
        raise ValueError("invalid/inordinately off-policy ratio for PPO-Clip smoke")
    ratio = math.exp(diff)
    clipped = min(1 + clip_epsilon, max(1 - clip_epsilon, ratio))
    return min(ratio * advantage, clipped * advantage)


def normalize_advantages(adv: Sequence[float], epsilon: float = 1e-8) -> list[float]:
    if not adv or epsilon <= 0 or not all(math.isfinite(float(x)) for x in adv):
        raise ValueError("invalid advantage normalization inputs")
    mu = sum(adv) / len(adv)
    std = math.sqrt(sum((x - mu) ** 2 for x in adv) / len(adv))
    if std < epsilon:
        return [0.0] * len(adv)
    return [(x - mu) / (std + epsilon) for x in adv]
