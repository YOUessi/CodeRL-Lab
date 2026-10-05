import pytest

from coderl_lab.execution import PythonExecutor, normalize_stdout
from coderl_lab.schema import StdIOTestCase, TestCase as CodeTestCase


def test_trusted_local_function_executor() -> None:
    executor = PythonExecutor(mode="local", allow_unsafe_local=True)
    report = executor.run(
        "def add(a, b):\n    return a + b",
        task_mode="function",
        entry_point="add",
        cases=[
            CodeTestCase(name="one", args=[1, 2], expected=3),
            CodeTestCase(name="two", args=[-1, 1], expected=0),
        ],
    )
    assert report.syntax_ok
    assert report.pass_rate == 1.0


def test_trusted_local_stdio_executor() -> None:
    executor = PythonExecutor(mode="local", allow_unsafe_local=True)
    report = executor.run(
        "a, b = map(int, input().split())\nprint(a + b)",
        task_mode="stdin_stdout",
        entry_point=None,
        cases=[
            StdIOTestCase(
                name="positive",
                stdin="1 2\n",
                expected_stdout="3\n",
            ),
            StdIOTestCase(
                name="negative",
                stdin="-5 2\n",
                expected_stdout="-3\n",
            ),
        ],
    )
    assert report.syntax_ok
    assert report.pass_rate == 1.0


def test_stdio_output_normalization_ignores_line_endings_and_trailing_space() -> None:
    assert normalize_stdout("3  \r\n\r\n") == "3"
    assert normalize_stdout("\n3\n") == "3"


def test_stdio_wrong_answer_is_detected() -> None:
    executor = PythonExecutor(mode="local", allow_unsafe_local=True)
    report = executor.run(
        "print(0)",
        task_mode="stdin_stdout",
        entry_point=None,
        cases=[
            StdIOTestCase(
                name="wrong",
                stdin="1 2\n",
                expected_stdout="3\n",
            )
        ],
    )
    assert report.syntax_ok
    assert report.pass_rate == 0.0
    assert report.cases[0].actual == "0"
    assert report.cases[0].expected == "3"


def test_syntax_error_fails_closed() -> None:
    executor = PythonExecutor(mode="local", allow_unsafe_local=True)
    report = executor.run(
        "def broken(:\n    pass",
        task_mode="function",
        entry_point="broken",
        cases=[CodeTestCase(name="syntax", args=[], expected=None)],
    )
    assert not report.syntax_ok
    assert report.pass_rate == 0.0


def test_function_mode_rejects_stdio_cases() -> None:
    executor = PythonExecutor(mode="local", allow_unsafe_local=True)
    with pytest.raises(TypeError):
        executor.run(
            "def solve():\n    return 1",
            task_mode="function",
            entry_point="solve",
            cases=[StdIOTestCase(stdin="", expected_stdout="1")],
        )
