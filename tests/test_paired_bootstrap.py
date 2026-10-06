from coderl_lab.analysis.paired_bootstrap import compare_summaries


def make_summary(counts):
    return {
        "per_task": {
            f"t{i}": {"samples": 4, "correct": c}
            for i, c in enumerate(counts)
        }
    }


def test_identical_models_have_zero_delta_ci() -> None:
    a = make_summary([0, 1, 2, 4])
    out = compare_summaries(a, a, iterations=1000, seed=1)
    p1 = out["pass_at_1_task_rate_delta"]
    p4 = out["pass_at_4_solved_task_delta"]
    assert p1["observed_delta"] == 0.0
    assert p1["ci95_low"] == 0.0
    assert p1["ci95_high"] == 0.0
    assert p4["observed_delta"] == 0.0
