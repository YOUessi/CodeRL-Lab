from coderl_lab.analysis.targeted_k_recovery import analyze_targeted_recovery


def summary(rows):
    return {
        "per_task": {
            task_id: {"samples": 64, "correct": correct}
            for task_id, correct in rows.items()
        }
    }


def test_targeted_recovery() -> None:
    out = analyze_targeted_recovery(
        original_targets=["a", "b", "c"],
        base=summary({"a": 4, "b": 2, "c": 1}),
        sft=summary({"a": 1, "b": 0, "c": 0}),
        grpo=summary({"a": 2, "b": 1, "c": 0}),
    )
    assert out["sft_recovered_at_64"] == 1
    assert out["sft_still_missing_at_64"] == 2
    assert out["grpo_recovers_sft_missing_at_64"] == 1
    assert out["sft_still_missing_task_ids"] == ["b", "c"]
