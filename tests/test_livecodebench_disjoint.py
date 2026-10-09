from __future__ import annotations

from copy import deepcopy

import pytest

from coderl_lab.datasets.livecodebench_disjoint import make_disjoint_audit


def _task(prefix: str, i: int):
    return {
        "question_id": f"{prefix}_{i}",
        "question_content": f"Compute number {prefix} {i} uniquely.",
        "starter_code": "",
        "public_test_cases": [{"input": "1", "output": "2", "testtype": "stdin"}],
    }


def test_disjoint_corpus_keeps_every_new_task() -> None:
    v5 = [_task("older", i) for i in range(167)]
    v6 = [_task("newer", i) for i in range(175)]
    filtered, audit = make_disjoint_audit(v5_rows=v5, v6_rows=v6)
    assert len(filtered) == 167
    assert audit["excluded_total"] == 0
    assert not audit["private_cases_accessed"]
    assert not audit["success_metrics_used_to_select_tasks"]


def test_overlap_is_removed_by_id_and_normalized_statement() -> None:
    v5 = [_task("older", i) for i in range(167)]
    v6 = [_task("newer", i) for i in range(175)]
    v5[0]["question_id"] = v6[0]["question_id"]
    v5[1]["question_content"] = (
        "  " + v6[1]["question_content"].upper().replace(" ", " \n ") + "  "
    )
    filtered, audit = make_disjoint_audit(v5_rows=v5, v6_rows=v6)
    assert len(filtered) == 165
    assert audit["excluded"] == {
        "same_normalized_problem_text": 1,
        "same_question_id": 1,
    }


def test_reject_duplicate_ids_and_private_data() -> None:
    v5 = [_task("older", i) for i in range(167)]
    v6 = [_task("newer", i) for i in range(175)]
    v5[1]["question_id"] = v5[0]["question_id"]
    with pytest.raises(ValueError, match="duplicate question ID"):
        make_disjoint_audit(v5_rows=v5, v6_rows=v6)

    v5[1]["question_id"] = "older_1"
    v6[0]["private_test_cases"] = "hidden"
    with pytest.raises(ValueError, match="private data"):
        make_disjoint_audit(v5_rows=v5, v6_rows=v6)


def test_reject_partial_v5_slice() -> None:
    v5 = [_task("older", i) for i in range(166)]
    v6 = [_task("newer", i) for i in range(175)]
    with pytest.raises(ValueError, match="expected v5 167"):
        make_disjoint_audit(v5_rows=v5, v6_rows=v6)
