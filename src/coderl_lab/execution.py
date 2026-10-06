from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .schema import TestCase


_RUNNER = r"""
import importlib.util
import json
import sys
import traceback

solution_path = sys.argv[1]
case = json.loads(sys.stdin.read())

try:
    spec = importlib.util.spec_from_file_location("candidate_solution", solution_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    namespace = module.__dict__
    setup_code = case.get("setup_code", "")
    if setup_code:
        exec(setup_code, namespace, namespace)

    assertion = case.get("assertion")
    if assertion:
        exec(assertion, namespace, namespace)
        actual = True
        expected = True
        passed = True
    else:
        fn = getattr(module, case["entry_point"])
        actual = fn(*case.get("args", []), **case.get("kwargs", {}))
        expected = case.get("expected")
        passed = actual == expected

    print(json.dumps({
        "passed": passed,
        "actual": actual,
        "expected": expected,
        "error": None,
    }, ensure_ascii=False, default=repr))
except BaseException:
    print(json.dumps({
        "passed": False,
        "actual": None,
        "expected": case.get("expected"),
        "error": traceback.format_exc(limit=8),
    }, ensure_ascii=False, default=repr))
"""


@dataclass(frozen=True)
class CaseResult:
    name: str
    passed: bool
    actual: object | None
    expected: object | None
    error: str | None
    timed_out: bool = False


@dataclass(frozen=True)
class ExecutionReport:
    syntax_ok: bool
    cases: tuple[CaseResult, ...]

    @property
    def passed(self) -> int:
        return sum(case.passed for case in self.cases)

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


def check_syntax(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


class PythonExecutor:
    """Execute candidate code inside the configured runtime."""

    def __init__(
        self,
        *,
        mode: str = "docker",
        timeout_seconds: float = 3.0,
        docker_image: str = "python:3.11-slim",
        memory_limit: str = "512m",
        allow_unsafe_local: bool = False,
    ) -> None:
        if mode not in {"docker", "local"}:
            raise ValueError("mode must be 'docker' or 'local'")
        if mode == "local" and not allow_unsafe_local:
            raise ValueError(
                "local execution is unsafe for untrusted model output; "
                "pass allow_unsafe_local=True only for trusted fixtures"
            )

        self.mode = mode
        self.timeout_seconds = timeout_seconds
        self.docker_image = docker_image
        self.memory_limit = memory_limit

        if self.mode == "docker":
            self._validate_docker_runtime()

    def _validate_docker_runtime(self) -> None:
        if shutil.which("docker") is None:
            raise RuntimeError("docker executable not found")

        inspect = subprocess.run(
            ["docker", "image", "inspect", self.docker_image],
            text=True,
            capture_output=True,
            check=False,
        )
        if inspect.returncode != 0:
            raise RuntimeError(
                f"Docker image '{self.docker_image}' is not available locally. "
                f"Pull it before evaluation: docker pull {self.docker_image}"
            )

    def run(
        self,
        code: str,
        entry_point: str,
        cases: Iterable[TestCase],
        setup_code: str = "",
    ) -> ExecutionReport:
        case_list = tuple(cases)
        if not check_syntax(code):
            return ExecutionReport(
                syntax_ok=False,
                cases=tuple(
                    CaseResult(
                        name=case.name,
                        passed=False,
                        actual=None,
                        expected=case.expected,
                        error="SyntaxError",
                    )
                    for case in case_list
                ),
            )

        with tempfile.TemporaryDirectory(prefix="coderl_lab_") as tmp:
            workdir = Path(tmp)
            # Rootless Docker needs execute permission to traverse the bind mount.
            workdir.chmod(0o755)

            solution_path = workdir / "solution.py"
            runner_path = workdir / "runner.py"
            solution_path.write_text(code, encoding="utf-8")
            runner_path.write_text(_RUNNER, encoding="utf-8")
            solution_path.chmod(0o444)
            runner_path.chmod(0o444)

            results = tuple(
                self._run_case(
                    workdir=workdir,
                    entry_point=entry_point,
                    case=case,
                    setup_code=setup_code,
                )
                for case in case_list
            )
        return ExecutionReport(syntax_ok=True, cases=results)

    def _run_case(
        self,
        *,
        workdir: Path,
        entry_point: str,
        case: TestCase,
        setup_code: str,
    ) -> CaseResult:
        payload = json.dumps(
            {
                "entry_point": entry_point,
                "args": case.args,
                "kwargs": case.kwargs,
                "expected": case.expected,
                "assertion": case.assertion,
                "setup_code": setup_code,
            },
            ensure_ascii=False,
        )

        container_name: str | None = None
        if self.mode == "docker":
            container_name = f"coderl-lab-{uuid.uuid4().hex[:12]}"
            command = [
                "docker",
                "run",
                "--rm",
                "--name",
                container_name,
                "--label",
                "coderl_lab=1",
                "--network",
                "none",
                "--memory",
                self.memory_limit,
                "--cpus",
                "1",
                "--pids-limit",
                "64",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "-i",
                "-v",
                f"{workdir}:/work:ro",
                "-w",
                "/work",
                self.docker_image,
                "python",
                "runner.py",
                "solution.py",
            ]
        else:
            command = [
                sys.executable,
                str(workdir / "runner.py"),
                str(workdir / "solution.py"),
            ]

        try:
            proc = subprocess.run(
                command,
                input=payload,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            if container_name is not None:
                subprocess.run(
                    ["docker", "rm", "-f", container_name],
                    text=True,
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
            return CaseResult(
                name=case.name,
                passed=False,
                actual=None,
                expected=case.expected,
                error="execution timed out",
                timed_out=True,
            )

        stdout = proc.stdout.strip().splitlines()
        if not stdout:
            return CaseResult(
                name=case.name,
                passed=False,
                actual=None,
                expected=case.expected,
                error=(proc.stderr.strip() or f"runner exited with {proc.returncode}"),
            )

        try:
            data = json.loads(stdout[-1])
        except json.JSONDecodeError:
            return CaseResult(
                name=case.name,
                passed=False,
                actual=None,
                expected=case.expected,
                error=f"invalid runner output: {stdout[-1][:300]}",
            )

        return CaseResult(
            name=case.name,
            passed=bool(data.get("passed")),
            actual=data.get("actual"),
            expected=data.get("expected"),
            error=data.get("error"),
            timed_out=False,
        )


def report_to_dict(report: ExecutionReport) -> dict:
    return {
        "syntax_ok": report.syntax_ok,
        "passed": report.passed,
        "total": report.total,
        "pass_rate": report.pass_rate,
        "cases": [asdict(case) for case in report.cases],
    }
