import pytest

from coderl_lab.schema import CodeTask, FunctionTestCase, StdIOTestCase


def valid_function_task() -> dict:
    return {
        "task_id": "demo",
        "prompt": "demo",
        "entry_point": "solve",
        "public_tests": [{"args": [1], "expected": 1}],
        "hidden_tests": [{"args": [2], "expected": 2}],
    }


def valid_stdio_task() -> dict:
    return {
        "task_id": "stdio-demo",
        "prompt": "读取两个整数并输出它们的和。",
        "task_mode": "stdin_stdout",
        "public_tests": [{"stdin": "1 2\n", "expected_stdout": "3\n"}],
        "hidden_tests": [{"stdin": "-1 5\n", "expected_stdout": "4\n"}],
    }


def test_legacy_task_defaults_to_function_mode() -> None:
    task = CodeTask.from_dict(valid_function_task())
    assert task.task_mode == "function"
    assert task.entry_point == "solve"
    assert isinstance(task.public_tests[0], FunctionTestCase)
    assert len(task.public_tests) == 1
    assert len(task.hidden_tests) == 1


def test_task_schema_accepts_stdio_mode_without_entry_point() -> None:
    task = CodeTask.from_dict(valid_stdio_task())
    assert task.task_mode == "stdin_stdout"
    assert task.entry_point is None
    assert isinstance(task.public_tests[0], StdIOTestCase)
    assert task.public_tests[0].stdin == "1 2\n"
    assert task.public_tests[0].expected_stdout == "3\n"


def test_function_task_requires_entry_point() -> None:
    data = valid_function_task()
    del data["entry_point"]
    with pytest.raises(ValueError, match="entry_point"):
        CodeTask.from_dict(data)


def test_task_requires_hidden_tests() -> None:
    data = valid_function_task()
    data["hidden_tests"] = []
    with pytest.raises(ValueError):
        CodeTask.from_dict(data)


def test_stdio_case_requires_string_input_and_output() -> None:
    data = valid_stdio_task()
    data["public_tests"] = [{"stdin": [1, 2], "expected_stdout": "3"}]
    with pytest.raises(TypeError):
        CodeTask.from_dict(data)


def test_rejects_unknown_task_mode() -> None:
    data = valid_function_task()
    data["task_mode"] = "agent"
    with pytest.raises(ValueError, match="task_mode"):
        CodeTask.from_dict(data)
