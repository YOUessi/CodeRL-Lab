import pytest

from coderl_lab.analysis.validate_large_k_artifacts import (
    validate_predictions,
    validate_three,
)


def rows(tasks=2, samples=3):
    return [
        {
            "task_id": f"t{task}",
            "sample_id": sample,
            "completion": "x",
        }
        for task in range(tasks)
        for sample in range(samples)
    ]


def test_validate_predictions_accepts_exact_grid() -> None:
    out = validate_predictions(
        rows(),
        expected_tasks=2,
        expected_samples=3,
    )
    assert out["rows"] == 6


def test_validate_predictions_rejects_missing_sample() -> None:
    bad = rows()
    bad.pop()
    with pytest.raises(ValueError):
        validate_predictions(
            bad,
            expected_tasks=2,
            expected_samples=3,
        )


def test_validate_three_requires_matching_task_sets() -> None:
    a = rows()
    b = rows()
    c = rows()
    c[-1]["task_id"] = "different"
    with pytest.raises(ValueError):
        validate_three(
            base=a,
            sft=b,
            grpo=c,
            expected_tasks=2,
            expected_samples=3,
        )
