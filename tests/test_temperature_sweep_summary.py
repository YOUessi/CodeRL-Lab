import pytest

from coderl_lab.analysis.temperature_sweep_summary import paired_task_bootstrap


def test_paired_task_bootstrap_positive_delta() -> None:
    a = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4}
    b = {"a": 0.2, "b": 0.3, "c": 0.4, "d": 0.5}
    out = paired_task_bootstrap(a, b, iterations=2000, seed=42)
    assert out["observed_delta"] == pytest.approx(0.1)
    assert out["ci95_low"] > 0
    assert out["bootstrap_probability_positive"] > 0.99
