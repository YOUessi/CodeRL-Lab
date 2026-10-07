from coderl_lab.analysis.local_bottleneck_statistics import summarize


def row(task, baseline, stable, unstable, high, eligible=True):
    return {
        "task_id": task,
        "eligible": eligible,
        "baseline": {"exact_match_reference": baseline},
        "low_stabilize": {
            "exact_match_reference": stable,
            "intervention_applied": eligible,
        },
        "low_destabilize": {
            "exact_match_reference": unstable,
            "intervention_applied": eligible,
        },
        "high_stabilize": {
            "exact_match_reference": high,
            "intervention_applied": eligible,
        },
    }


def test_summary_detects_rescue_and_induced_divergence() -> None:
    arms = [
        {
            "per_task": {
                "a": row("a", False, True, False, False),
                "b": row("b", True, True, False, True),
            }
        },
        {
            "per_task": {
                "a": row("a", False, True, False, False),
                "b": row("b", True, True, True, True),
            }
        },
    ]
    out = summarize(arms, iterations=100, seed=42)
    assert out["eligible_tasks"] == 2
    assert out["rescue"]["baseline_divergent_pairs"] == 2
    assert out["rescue"]["low_stabilize_rescued"] == 2
    assert out["rescue"]["high_stabilize_rescued"] == 0
    assert out["induced_divergence"]["baseline_matching_pairs"] == 2
    assert out["induced_divergence"]["low_destabilize_induced"] == 1
