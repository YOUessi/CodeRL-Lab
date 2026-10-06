from coderl_lab.analysis.track_k4_losses import track_k4_losses


def summary(rows, samples):
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


def test_track_k4_losses() -> None:
    base_k4 = summary({"a": 1, "b": 2, "c": 0}, 4)
    sft_k4 = summary({"a": 0, "b": 0, "c": 1}, 4)
    base_n16 = summary({"a": 2, "b": 3, "c": 0}, 16)
    sft_n16 = summary({"a": 1, "b": 0, "c": 2}, 16)
    grpo_n16 = summary({"a": 1, "b": 1, "c": 1}, 16)

    out = track_k4_losses(
        base_k4=base_k4,
        sft_k4=sft_k4,
        base_n16=base_n16,
        sft_n16=sft_n16,
        grpo_n16=grpo_n16,
    )

    assert out["k4_lost_task_count"] == 2
    assert out["sft_recovered_at_n16"] == 1
    assert out["sft_still_missing_at_n16"] == 1
    assert out["grpo_recovers_sft_missing_at_n16"] == 1
    assert out["sft_still_missing_task_ids"] == ["b"]
