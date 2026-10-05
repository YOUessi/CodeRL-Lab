from coderl_lab.execution import PythonExecutor
from coderl_lab.schema import TestCase as CodeTestCase


def test_trusted_local_executor() -> None:
    executor = PythonExecutor(mode="local", allow_unsafe_local=True)
    report = executor.run(
        "def add(a, b):\n    return a + b",
        entry_point="add",
        cases=[
            CodeTestCase(name="one", args=[1, 2], expected=3),
            CodeTestCase(name="two", args=[-1, 1], expected=0),
        ],
    )
    assert report.syntax_ok
    assert report.pass_rate == 1.0


def test_syntax_error_fails_closed() -> None:
    executor = PythonExecutor(mode="local", allow_unsafe_local=True)
    report = executor.run(
        "def broken(:\n    pass",
        entry_point="broken",
        cases=[CodeTestCase(name="syntax", args=[], expected=None)],
    )
    assert not report.syntax_ok
    assert report.pass_rate == 0.0
