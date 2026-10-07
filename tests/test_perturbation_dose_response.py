from coderl_lab.analysis.perturbation_dose_response import (
    SCALE_MAP,
    scale_and_seed,
)


def test_scale_and_seed_parsing() -> None:
    assert scale_and_seed("scale025_seed101") == (0.25, 101)
    assert scale_and_seed("scale050_seed202") == (0.5, 202)
    assert scale_and_seed("scale100_seed303") == (1.0, 303)
    assert scale_and_seed("scale200_seed101") == (2.0, 101)


def test_scale_map_is_monotonic() -> None:
    values = [SCALE_MAP[key] for key in ("025", "050", "100", "200")]
    assert values == [0.25, 0.5, 1.0, 2.0]
