from coderl_lab.analysis.bottleneck_gated_policy import summarize_gated_policy


def test_public_fail_uses_destabilize() -> None:
    rows = [
        {
            "task_id": "a",
            "baseline_correct": False,
            "low_destabilize_correct": True,
        },
        {
            "task_id": "b",
            "baseline_correct": True,
            "low_destabilize_correct": False,
        },
    ]
    reference_eval = {
        "a": {"public": {"pass_rate": 0.0}},
        "b": {"public": {"pass_rate": 1.0}},
    }

    out = summarize_gated_policy(
        rows,
        reference_eval,
        iterations=200,
        seed=42,
    )
    assert out["baseline_correct_fraction"] == 0.5
    assert out["gated_correct_fraction"] == 1.0
    assert out["transitions"]["wrong_to_correct"] == 1
    assert out["transitions"]["correct_to_wrong"] == 0
