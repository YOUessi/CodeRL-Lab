from coderl_lab.analysis.local_bottleneck_intervention import (
    select_intervention_positions,
)


def test_selects_first_low_margin_and_nearest_high_control() -> None:
    low, high = select_intervention_positions(
        [0.4, 0.04, 0.1, 0.3],
        primary_window=4,
        low_threshold=0.05,
        high_threshold=0.2,
    )
    assert low == 1
    assert high == 0


def test_no_low_margin_is_ineligible() -> None:
    low, high = select_intervention_positions(
        [0.2, 0.3, 0.4],
        primary_window=3,
    )
    assert low is None
    assert high is None


def test_window_is_respected() -> None:
    low, high = select_intervention_positions(
        [0.4, 0.3, 0.02],
        primary_window=2,
    )
    assert low is None
    assert high is None
