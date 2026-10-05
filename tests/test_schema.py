import pytest

from coderl_lab.schema import CodeTask


def valid_task() -> dict:
    return {
        "task_id": "demo",
        "prompt": "demo",
        "entry_point": "solve",
        "public_tests": [{"args": [1], "expected": 1}],
        "hidden_tests": [{"args": [2], "expected": 2}],
    }


def test_task_schema_accepts_valid_task() -> None:
    task = CodeTask.from_dict(valid_task())
    assert task.task_id == "demo"
    assert len(task.public_tests) == 1
    assert len(task.hidden_tests) == 1


def test_task_requires_hidden_tests() -> None:
    data = valid_task()
    data["hidden_tests"] = []
    with pytest.raises(ValueError):
        CodeTask.from_dict(data)
