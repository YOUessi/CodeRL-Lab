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


def test_docker_timeout_cleanup_command(monkeypatch, tmp_path) -> None:
    import subprocess

    calls = []

    class Done:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[:3] == ["docker", "image", "inspect"]:
            return Done()
        if command[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(command, timeout=1)
        if command[:3] == ["docker", "rm", "-f"]:
            return Done()
        return Done()

    monkeypatch.setattr("coderl_lab.execution.shutil.which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr("coderl_lab.execution.subprocess.run", fake_run)

    executor = PythonExecutor(mode="docker", timeout_seconds=1)
    report = executor.run(
        "def loop():\n    while True:\n        pass",
        entry_point="loop",
        cases=[CodeTestCase(name="timeout", args=[], expected=None)],
    )

    assert report.cases[0].timed_out
    run_command = next(cmd for cmd in calls if cmd[:2] == ["docker", "run"])
    name_index = run_command.index("--name") + 1
    container_name = run_command[name_index]
    assert container_name.startswith("coderl-lab-")
    assert "--label" in run_command
    assert ["docker", "rm", "-f", container_name] in calls
