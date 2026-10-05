import pytest

from coderl_lab.metrics import estimate_pass_at_k, mean_pass_at_k


def test_pass_at_1() -> None:
    assert estimate_pass_at_k(n=10, c=2, k=1) == pytest.approx(0.2)


def test_pass_at_k_is_one_when_k_exceeds_failures() -> None:
    assert estimate_pass_at_k(n=4, c=2, k=3) == 1.0


def test_mean_pass_at_k() -> None:
    value = mean_pass_at_k([(4, 2), (4, 1)], k=1)
    assert value == pytest.approx((0.5 + 0.25) / 2)


@pytest.mark.parametrize(
    "n,c,k",
    [(0, 0, 1), (3, 4, 1), (3, -1, 1), (3, 1, 0), (3, 1, 4)],
)
def test_invalid_arguments(n: int, c: int, k: int) -> None:
    with pytest.raises(ValueError):
        estimate_pass_at_k(n=n, c=c, k=k)
