from coderl_lab.analysis.margin_shift_statistics import paired_mean_bootstrap


def test_paired_mean_bootstrap_positive_shift() -> None:
    out = paired_mean_bootstrap(
        [0.0, 0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3, 0.4],
        iterations=1000,
        seed=42,
    )
    assert out["observed_mean_delta_sft_minus_base"] > 0
    assert out["ci95_low"] > 0
    assert out["tasks_increased"] == 4
    assert out["tasks_decreased"] == 0
