from coderl_lab.execution import PythonExecutor
from coderl_lab.schema import TestCase as CodeTestCase


def test_assertion_case_passes() -> None:
    executor = PythonExecutor(mode="local", allow_unsafe_local=True)
    report = executor.run(
        "def square(x):\n    return x * x",
        entry_point="square",
        cases=[
            CodeTestCase(
                name="assert_square",
                assertion="assert square(4) == 16",
            )
        ],
    )
    assert report.pass_rate == 1.0


def test_assertion_case_fails() -> None:
    executor = PythonExecutor(mode="local", allow_unsafe_local=True)
    report = executor.run(
        "def square(x):\n    return x + x",
        entry_point="square",
        cases=[
            CodeTestCase(
                name="assert_square",
                assertion="assert square(4) == 16",
            )
        ],
    )
    assert report.pass_rate == 0.0
    assert "AssertionError" in (report.cases[0].error or "")


def test_setup_code_is_available_to_assertion() -> None:
    executor = PythonExecutor(mode="local", allow_unsafe_local=True)
    report = executor.run(
        "def root(x):\n    return sqrt(x)",
        entry_point="root",
        setup_code="from math import sqrt",
        cases=[
            CodeTestCase(
                name="assert_root",
                assertion="assert root(9) == 3",
            )
        ],
    )
    assert report.pass_rate == 1.0
