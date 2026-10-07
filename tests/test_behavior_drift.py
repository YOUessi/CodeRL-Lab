import pytest

from coderl_lab.analysis.behavior_drift import compare_behavior


def row(task, sample, completion):
    return {
        "task_id": task,
        "sample_id": sample,
        "completion": completion,
        "raw_completion": completion,
    }


def test_identical_behavior() -> None:
    rows = [row("a", 0, "x"), row("a", 1, "y")]
    out = compare_behavior(rows, list(rows))
    assert out["exact_completion_match_fraction"] == 1.0
    assert out["changed_rows"] == 0
    assert out["changed_tasks"] == 0


def test_behavior_drift_detected() -> None:
    ref = [
        row("a", 0, "x"),
        row("a", 1, "x"),
        row("b", 0, "m"),
        row("b", 1, "n"),
    ]
    cand = [
        row("a", 0, "x"),
        row("a", 1, "z"),
        row("b", 0, "m"),
        row("b", 1, "n"),
    ]
    out = compare_behavior(ref, cand)
    assert out["rows_compared"] == 4
    assert out["exact_completion_matches"] == 3
    assert out["changed_rows"] == 1
    assert out["changed_tasks"] == 1
    assert out["changed_task_ids"] == ["a"]
    assert (
        out["candidate_diversity"]["unique_fraction"]
        > out["reference_diversity"]["unique_fraction"]
    )


def test_prediction_key_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        compare_behavior([row("a", 0, "x")], [row("b", 0, "x")])
