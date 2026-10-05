from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingReward:
    syntax: float
    public_pass_rate: float
    all_public_pass_bonus: float
    total: float


def compute_training_reward(
    *,
    syntax_ok: bool,
    public_pass_rate: float,
    syntax_weight: float = 0.10,
    public_test_weight: float = 0.80,
    all_public_pass_bonus_weight: float = 0.10,
) -> TrainingReward:
    """Compute reward using training-visible signals only.

    Hidden tests are intentionally excluded. They are reserved for evaluation
    and must never leak into the training reward.
    """
    if not 0.0 <= public_pass_rate <= 1.0:
        raise ValueError("public_pass_rate must be in [0, 1]")

    weights = syntax_weight + public_test_weight + all_public_pass_bonus_weight
    if abs(weights - 1.0) > 1e-9:
        raise ValueError("reward weights must sum to 1.0")

    syntax = 1.0 if syntax_ok else 0.0
    all_public = 1.0 if syntax_ok and public_pass_rate == 1.0 else 0.0
    total = (
        syntax_weight * syntax
        + public_test_weight * public_pass_rate
        + all_public_pass_bonus_weight * all_public
    )
    return TrainingReward(
        syntax=syntax,
        public_pass_rate=public_pass_rate,
        all_public_pass_bonus=all_public,
        total=total,
    )
