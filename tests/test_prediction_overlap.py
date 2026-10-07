from coderl_lab.analysis.prediction_overlap import analyze


def test_prediction_overlap_counts_changed_rows() -> None:
    a = [
        {"task_id": "a", "sample_id": 0, "completion": "x", "raw_completion": "x"},
        {"task_id": "a", "sample_id": 1, "completion": "y", "raw_completion": "y"},
        {"task_id": "b", "sample_id": 0, "completion": "z", "raw_completion": "z"},
    ]
    b = [
        {"task_id": "a", "sample_id": 0, "completion": "x", "raw_completion": "x"},
        {"task_id": "a", "sample_id": 1, "completion": "q", "raw_completion": "q"},
        {"task_id": "b", "sample_id": 0, "completion": "z", "raw_completion": "raw-z"},
    ]

    result = analyze(a, b)

    assert result["rows_compared"] == 3
    assert result["completion_matches"] == 2
    assert result["raw_matches"] == 1
    assert result["changed_rows"] == 1
    assert result["changed_tasks"] == 1
