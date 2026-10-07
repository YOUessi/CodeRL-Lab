import pytest

from coderl_lab.analysis.sampling_amplification_summary import parse_arm


def test_parse_sampling_amplification_arm() -> None:
    assert parse_arm("scale025_seed101") == (0.25, 101)
    assert parse_arm("scale050_seed202") == (0.5, 202)
    assert parse_arm("scale100_seed303") == (1.0, 303)
    assert parse_arm("scale200_seed101") == (2.0, 101)


def test_invalid_sampling_amplification_arm() -> None:
    with pytest.raises(ValueError):
        parse_arm("scale050_seed999")
