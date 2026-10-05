import json

import pytest

from coderl_lab.dataset.apps import (
    AppsAdapterConfig,
    AppsAdapterError,
    convert_apps_row,
)
from coderl_lab.schema import CodeTask, FunctionTestCase, StdIOTestCase


def function_row() -> dict:
    return {
        "problem_id": 7,
        "question": "Return the sum.",
        "solutions": json.dumps(["def add(a, b):\n    return a + b"]),
        "input_output": json.dumps(
            {
                "fn_name": "add",
                "inputs": [[1, 2], [-1, 5], [0, 0], [10, 20]],
                "outputs": [3, 4, 0, 30],
            }
        ),
        "difficulty": "introductory",
        "url": "https://example.test/7",
        "starter_code": "def add(a, b):\n    pass",
    }


def stdio_row() -> dict:
    return {
        "problem_id": 8,
        "question": "Read two integers and print the sum.",
        "solutions": json.dumps(
            ["a, b = map(int, input().split())\nprint(a + b)"]
        ),
        "input_output": json.dumps(
            {
                "inputs": [
                    "1 2\n",
                    "-1 5\n",
                    "0 0\n",
                    "10 20\n",
                    "3 9\n",
                    "4 8\n",
                ],
                "outputs": ["3\n", "4\n", "0\n", "30\n", "12\n", "12\n"],
            }
        ),
        "difficulty": "introductory",
        "url": "https://example.test/8",
        "starter_code": "",
    }


def test_converts_function_task_and_keeps_reference_solution() -> None:
    result = convert_apps_row(function_row(), split="train")
    task = CodeTask.from_dict(result)

    assert task.task_mode == "function"
    assert task.entry_point == "add"
    assert isinstance(task.public_tests[0], FunctionTestCase)
    assert len(task.public_tests) == 1
    assert len(task.hidden_tests) == 3
    assert len(task.reference_solutions) == 1


def test_converts_stdio_task() -> None:
    result = convert_apps_row(stdio_row(), split="train")
    task = CodeTask.from_dict(result)

    assert task.task_mode == "stdin_stdout"
    assert task.entry_point is None
    assert isinstance(task.public_tests[0], StdIOTestCase)
    assert len(task.public_tests) == 2
    assert len(task.hidden_tests) == 4


def test_split_is_deterministic_for_same_seed() -> None:
    first = convert_apps_row(
        stdio_row(),
        split="train",
        config=AppsAdapterConfig(seed=123),
    )
    second = convert_apps_row(
        stdio_row(),
        split="train",
        config=AppsAdapterConfig(seed=123),
    )
    assert first["metadata"]["public_test_indices"] == second["metadata"][
        "public_test_indices"
    ]
    assert first["public_tests"] == second["public_tests"]


def test_rejects_too_few_tests() -> None:
    row = function_row()
    io = json.loads(row["input_output"])
    io["inputs"] = io["inputs"][:3]
    io["outputs"] = io["outputs"][:3]
    row["input_output"] = json.dumps(io)

    with pytest.raises(AppsAdapterError) as exc:
        convert_apps_row(row, split="train")
    assert exc.value.reason == "too_few_tests"


def test_rejects_ambiguous_stdio_structure() -> None:
    row = stdio_row()
    io = json.loads(row["input_output"])
    io["inputs"][0] = ["1 2", "unexpected second string"]
    row["input_output"] = json.dumps(io)

    with pytest.raises(AppsAdapterError) as exc:
        convert_apps_row(row, split="train")
    assert exc.value.reason == "unsupported_stdin_format"


def test_rejects_missing_reference_solution() -> None:
    row = function_row()
    row["solutions"] = "[]"
    with pytest.raises(AppsAdapterError) as exc:
        convert_apps_row(row, split="train")
    assert exc.value.reason == "missing_reference_solutions"


def test_accepts_raw_jsonl_id_field_and_records_revision() -> None:
    row = function_row()
    row["id"] = row.pop("problem_id")
    result = convert_apps_row(
        row,
        split="train",
        source_revision="revision-test",
    )
    assert result["task_id"] == "apps/train/7"
    assert result["metadata"]["source_revision"] == "revision-test"
