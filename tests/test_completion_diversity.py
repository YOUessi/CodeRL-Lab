import pytest

from coderl_lab.analysis.completion_diversity import summarize


def test_diversity_detects_concentration() -> None:
    rows = [
        {"task_id": "a", "completion": "x"},
        {"task_id": "a", "completion": "x"},
        {"task_id": "a", "completion": "y"},
        {"task_id": "a", "completion": "z"},
    ]
    out = summarize(rows)
    task = out["per_task"]["a"]
    assert task["unique_completions"] == 3
    assert task["unique_fraction"] == pytest.approx(0.75)
    assert task["modal_fraction"] == pytest.approx(0.5)
    assert task["duplicate_fraction"] == pytest.approx(0.25)
