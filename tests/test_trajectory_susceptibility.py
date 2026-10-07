import pytest

from coderl_lab.analysis.susceptibility_statistics import (
    quartile_contrast,
    spearman,
)
from coderl_lab.analysis.trajectory_margin_profile import _margin_summary


def test_margin_summary_thresholds() -> None:
    out = _margin_summary(
        [0.0, 0.01, 0.05, 0.20],
        [0.0, 0.1, 0.2, 1.0],
    )
    p = out["probability_margin"]
    assert out["token_count"] == 4
    assert p["fraction_le_0_01"] == pytest.approx(0.5)
    assert p["fraction_le_0_05"] == pytest.approx(0.75)
    assert p["fraction_le_0_10"] == pytest.approx(0.75)


def test_spearman_monotonic() -> None:
    assert spearman([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [8, 6, 4, 2]) == pytest.approx(-1.0)


def test_quartile_contrast() -> None:
    out = quartile_contrast(
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 0, 0, 0, 1, 1, 1, 1],
    )
    assert out["quartile_size"] == 2
    assert out["bottom_outcome_mean"] == 0
    assert out["top_outcome_mean"] == 1
    assert out["top_minus_bottom"] == 1
