from coderl_lab.analysis.large_k_extend import (
    prepare_extension,
    verify_prefix,
)


def test_prepare_extension_preserves_original_task_order_and_seed() -> None:
    support = {
        "task_ids": {
            "base_lost_by_sft": ["b", "d"]
        }
    }
    tasks = [
        {"task_id": "a"},
        {"task_id": "b"},
        {"task_id": "c"},
        {"task_id": "d"},
    ]
    predictions = [
        {"task_id": "b", "sample_id": 0, "completion": "x", "generation": {"seed": 100042}},
        {"task_id": "d", "sample_id": 0, "completion": "y", "generation": {"seed": 300042}},
    ]

    selected, seeds = prepare_extension(
        support_analysis=support,
        tasks=tasks,
        base_predictions=predictions,
    )
    assert [row["task_id"] for row in selected] == ["b", "d"]
    assert seeds == {"b": 100042, "d": 300042}


def test_verify_prefix_detects_identical_first_16() -> None:
    original = [
        {"task_id": "b", "sample_id": i, "completion": f"c{i}"}
        for i in range(16)
    ]
    extended = original + [
        {"task_id": "b", "sample_id": i, "completion": f"c{i}"}
        for i in range(16, 64)
    ]
    out = verify_prefix(
        original_predictions=original,
        extended_predictions=extended,
        target_ids={"b"},
        prefix_samples=16,
    )
    assert out["identical"]
    assert out["matched_rows"] == 16


def test_verify_prefix_detects_mismatch() -> None:
    original = [{"task_id": "b", "sample_id": 0, "completion": "a"}]
    extended = [{"task_id": "b", "sample_id": 0, "completion": "different"}]
    out = verify_prefix(
        original_predictions=original,
        extended_predictions=extended,
        target_ids={"b"},
        prefix_samples=16,
    )
    assert not out["identical"]
    assert out["mismatched_rows"] == [{"task_id": "b", "sample_id": 0}]
