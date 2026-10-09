import json
from pathlib import Path
import pytest

from coderl_lab.analysis.exp004n_public_shift import compare_v5_v6


def _record(qid: str, *, difficulty: str, platform: str, text: str):
    return {
        "question_id": qid, "question_title": qid,
        "question_content": text, "starter_code": "",
        "platform": platform, "difficulty": difficulty,
        "contest_date": "2025-01-01",
        "public_test_cases": [{"input":"1","output":"2","testtype":"stdin"}],
        "metadata": {},
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_public_shift_counts_metadata_without_private_or_outcomes(tmp_path: Path):
    v5, v6 = tmp_path/"v5.jsonl", tmp_path/"v6.jsonl"
    _write(v5, [_record("v5_a", difficulty="easy", platform="atcoder", text="Easy problem")])
    _write(v6, [
        _record("v6_a", difficulty="hard", platform="leetcode", text="Hard problem"),
        _record("v6_b", difficulty="hard", platform="atcoder", text="Harder problem"),
    ])
    result=compare_v5_v6(v5_public=v5, v6_public=v6, require_pinned=False)
    assert result["uses_private_tests"] is False
    assert result["uses_generated_answers"] is False
    assert result["v5"]["tasks"] == 1
    assert result["v6"]["tasks"] == 2
    assert result["v6_minus_v5_difficulty_fraction"]["hard"] == 1.0


def test_task_id_reuse_rejected(tmp_path: Path):
    a,b=tmp_path/"a",tmp_path/"b"
    _write(a,[_record("same",difficulty="easy",platform="leetcode",text="First question")])
    _write(b,[_record("same",difficulty="hard",platform="atcoder",text="Second question")])
    with pytest.raises(ValueError,match="task ID overlap"):
        compare_v5_v6(v5_public=a,v6_public=b,require_pinned=False)


def test_same_task_text_rejected_even_when_ids_differ(tmp_path: Path):
    a,b=tmp_path/"a",tmp_path/"b"
    _write(a,[_record("v5",difficulty="easy",platform="leetcode",text="Array sorted ascending")])
    _write(b,[_record("v6",difficulty="hard",platform="atcoder",text=" array  sorted  ASCENDING ")])
    with pytest.raises(ValueError,match="normalized"):
        compare_v5_v6(v5_public=a,v6_public=b,require_pinned=False)


def test_private_case_in_public_view_rejected(tmp_path: Path):
    a,b=tmp_path/"a",tmp_path/"b"
    row=_record("v5",difficulty="easy",platform="atcoder",text="Puzzle A")
    row["private_test_cases"]="SECRET"
    _write(a,[row])
    _write(b,[_record("v6",difficulty="hard",platform="leetcode",text="Puzzle B")])
    with pytest.raises(ValueError,match="private_test_cases"):
        compare_v5_v6(v5_public=a,v6_public=b,require_pinned=False)
