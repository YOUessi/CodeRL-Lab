import json

from coderl_lab.datasets.exp004p_public_v4 import canonical_key, prepare


def example(task_id, body="Problem", *, private="SECRET_PRIVATE_TEST"):
    return {
        "question_id": task_id,
        "question_title": "Example",
        "question_content": body,
        "platform": "AtCoder",
        "contest_id": "contest",
        "contest_date": "2024-01-01",
        "starter_code": "",
        "difficulty": "easy",
        "public_test_cases": json.dumps([{
            "input": "1", "output": "1", "testtype": "stdin"
        }]),
        "private_test_cases": private,
        "metadata": "{}",
    }


def test_public_only_excludes_used_ids_and_body_duplicates(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    groups = {
        "test4.jsonl": [
            example("keep", "unique"),
            example("old_id", "id collision"),
            example("dup_body", " same    wording "),
            example("keep2", "second"),
        ],
        "test5.jsonl": [example("old_id", "unrelated")],
        "test6.jsonl": [example("new_id", "same wording")],
    }
    for filename, rows in groups.items():
        (source / filename).write_text(
            "".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8"
        )
    out = tmp_path / "out"
    result = prepare({name: source / name for name in groups}, out)
    assert result["accepted_count"] == 2
    assert result["exclusions"] == {
        "used_normalized_statement": 1, "used_question_id": 1
    }
    assert result["private_tests_decoded"] is False
    exported = (out / "public_tasks.jsonl").read_text(encoding="utf-8")
    assert "SECRET_PRIVATE_TEST" not in exported
    assert "private_test_cases" not in exported
    assert [json.loads(x)["question_id"] for x in exported.splitlines()] == [
        "keep", "keep2"
    ]
    assert canonical_key({"question_content": "  A   B  "}) == canonical_key(
        {"question_content": "a b"}
    )
