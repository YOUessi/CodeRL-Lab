import pytest

from coderl_lab.analysis.mbppplus_eval import build_overlap, summarize


def test_overlap_requires_source_id_and_entry_match() -> None:
    tasks = [
        {
            "task_id": "mbpp_validation_0554",
            "entry_point": "Split",
            "metadata": {"source_task_id": 554},
        },
        {
            "task_id": "mbpp_validation_0555",
            "entry_point": "other",
            "metadata": {"source_task_id": 555},
        },
    ]
    plus = {
        "Mbpp/554": {
            "task_id": "Mbpp/554",
            "entry_point": "Split",
            "base_input": [1, 2, 3],
            "plus_input": list(range(20)),
        }
    }
    out = build_overlap(tasks, plus)
    assert len(out) == 1
    assert out[0]["source_task_id"] == 554
    assert out[0]["plus_tests"] == 20


def test_summary_requires_base_and_plus_for_robust_correctness() -> None:
    rows = [
        {
            "task_id": "Mbpp/1",
            "sample_id": 0,
            "base_correct": True,
            "plus_correct": True,
        },
        {
            "task_id": "Mbpp/1",
            "sample_id": 1,
            "base_correct": True,
            "plus_correct": False,
        },
        {
            "task_id": "Mbpp/2",
            "sample_id": 0,
            "base_correct": False,
            "plus_correct": False,
        },
        {
            "task_id": "Mbpp/2",
            "sample_id": 1,
            "base_correct": True,
            "plus_correct": True,
        },
    ]
    out = summarize(rows, (1, 2))
    assert out["base_pass_at_k"]["pass@1"] == pytest.approx(0.75)
    assert out["plus_pass_at_k"]["pass@1"] == pytest.approx(0.5)
    assert out["base_solved_tasks"] == 2
    assert out["plus_solved_tasks"] == 2
    assert out["mean_robustness_gap"] == pytest.approx(0.25)
