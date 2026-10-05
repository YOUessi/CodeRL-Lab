from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .schema import (
    AnyTestCase,
    FunctionTestCase,
    StdIOTestCase,
    TaskMode,
)


_FUNCTION_RUNNER = r"""
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


def normalize_stdout(text: str) -> str:
    """竞赛式输出的第一版确定性规范化规则。"""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _expected_value(case: AnyTestCase) -> object | None:
    if isinstance(case, FunctionTestCase):
        return case.expected
    return case.expected_stdout


class PythonExecutor:
    """执行函数调用或标准输入输出 Python 候选。

    模型生成代码默认使用 Docker。local 模式仅用于仓库内可信夹具，
    且必须显式开启 allow_unsafe_local。
    """

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
        *,
        entry_point: str | None,
        cases: Iterable[AnyTestCase],
        task_mode: TaskMode = "function",
    ) -> ExecutionReport:
        case_list = tuple(cases)
        self._validate_cases(task_mode, entry_point, case_list)

        if not check_syntax(code):
            return ExecutionReport(
                syntax_ok=False,
                cases=tuple(
                    CaseResult(
                        name=case.name,
                        passed=False,
                        actual=None,
                        expected=_expected_value(case),
                        error="SyntaxError",
                    )
                    for case in case_list
                ),
            )

        with tempfile.TemporaryDirectory(prefix="coderl_lab_") as tmp:
            workdir = Path(tmp)
            # Rootless Docker 需要能够遍历宿主临时目录；保持目录不可写并
            # 通过只读 bind mount 限制候选代码。
            workdir.chmod(0o755)

            solution_path = workdir / "solution.py"
            solution_path.write_text(code, encoding="utf-8")
            solution_path.chmod(0o444)

            if task_mode == "function":
                runner_path = workdir / "runner.py"
                runner_path.write_text(_FUNCTION_RUNNER, encoding="utf-8")
                runner_path.chmod(0o444)
                results = tuple(
                    self._run_function_case(
                        workdir=workdir,
                        entry_point=entry_point or "",
                        case=case,
                    )
                    for case in case_list
                    if isinstance(case, FunctionTestCase)
                )
            else:
                results = tuple(
                    self._run_stdio_case(
                        workdir=workdir,
                        case=case,
                    )
                    for case in case_list
                    if isinstance(case, StdIOTestCase)
                )

        return ExecutionReport(syntax_ok=True, cases=results)

    @staticmethod
    def _validate_cases(
        task_mode: TaskMode,
        entry_point: str | None,
        cases: tuple[AnyTestCase, ...],
    ) -> None:
        if task_mode == "function":
            if not entry_point:
                raise ValueError("function execution requires entry_point")
            if not all(isinstance(case, FunctionTestCase) for case in cases):
                raise TypeError("function task received a non-function test case")
        elif task_mode == "stdin_stdout":
            if not all(isinstance(case, StdIOTestCase) for case in cases):
                raise TypeError(
                    "stdin_stdout task received a non-stdin/stdout test case"
                )
        else:
            raise ValueError("unsupported task_mode")

    def _docker_prefix(self, workdir: Path) -> list[str]:
        return [
            "docker",
            "run",
            "--rm",
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
        ]

    def _run_function_case(
        self,
        *,
        workdir: Path,
        entry_point: str,
        case: FunctionTestCase,
    ) -> CaseResult:
        payload = json.dumps(
            {
                "entry_point": entry_point,
                "args": case.args,
                "kwargs": case.kwargs,
                "expected": case.expected,
            },
            ensure_ascii=False,
        )

        if self.mode == "docker":
            command = self._docker_prefix(workdir) + [
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

        return self._execute_function_process(command, payload, case)

    def _execute_function_process(
        self,
        command: list[str],
        payload: str,
        case: FunctionTestCase,
    ) -> CaseResult:
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

    def _run_stdio_case(
        self,
        *,
        workdir: Path,
        case: StdIOTestCase,
    ) -> CaseResult:
        if self.mode == "docker":
            command = self._docker_prefix(workdir) + [
                "python",
                "solution.py",
            ]
        else:
            command = [
                sys.executable,
                str(workdir / "solution.py"),
            ]

        try:
            proc = subprocess.run(
                command,
                input=case.stdin,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return CaseResult(
                name=case.name,
                passed=False,
                actual=None,
                expected=case.expected_stdout,
                error="execution timed out",
                timed_out=True,
            )

        actual = normalize_stdout(proc.stdout)
        expected = normalize_stdout(case.expected_stdout)

        if proc.returncode != 0:
            return CaseResult(
                name=case.name,
                passed=False,
                actual=actual,
                expected=expected,
                error=(
                    proc.stderr.strip()
                    or f"candidate exited with {proc.returncode}"
                ),
            )

        return CaseResult(
            name=case.name,
            passed=actual == expected,
            actual=actual,
            expected=expected,
            error=None,
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
