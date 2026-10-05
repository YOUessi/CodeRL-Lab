import pytest

from coderl_lab.reward import compute_training_reward


def test_full_public_pass_reward() -> None:
    reward = compute_training_reward(syntax_ok=True, public_pass_rate=1.0)
    assert reward.total == pytest.approx(1.0)


def test_partial_public_pass_reward() -> None:
    reward = compute_training_reward(syntax_ok=True, public_pass_rate=0.5)
    assert reward.total == pytest.approx(0.5)
    assert reward.all_public_pass_bonus == 0.0


def test_invalid_pass_rate() -> None:
    with pytest.raises(ValueError):
        compute_training_reward(syntax_ok=True, public_pass_rate=1.1)
