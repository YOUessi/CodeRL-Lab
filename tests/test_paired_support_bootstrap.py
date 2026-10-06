from coderl_lab.analysis.paired_support_bootstrap import compare_support


def summary(rows, samples=16):
    return {
        "per_task": {
            task_id: {
                "samples": samples,
                "correct": correct,
                "empirical_success_rate": correct / samples,
            }
            for task_id, correct in rows.items()
        }
    }


def test_compare_support_reports_sample_count_and_solved_delta() -> None:
    a = summary({"a": 1, "b": 0, "c": 2})
    b = summary({"a": 1, "b": 1, "c": 0})
    out = compare_support(a, b, iterations=1000, seed=7)
    assert out["samples_per_task"] == 16
    assert "solved_at_n_delta" in out
    assert "empirical_success_rate_delta" in out
