from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias


TaskMode = Literal["function", "stdin_stdout"]


@dataclass(frozen=True)
class FunctionTestCase:
    """函数调用模式测试用例。"""

    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    expected: Any = None
    name: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FunctionTestCase":
        if not isinstance(data, dict):
            raise TypeError("test case must be an object")
        args = data.get("args", [])
        kwargs = data.get("kwargs", {})
        if not isinstance(args, list):
            raise TypeError("function test case 'args' must be a list")
        if not isinstance(kwargs, dict):
            raise TypeError("function test case 'kwargs' must be an object")
        return cls(
            args=args,
            kwargs=kwargs,
            expected=data.get("expected"),
            name=str(data.get("name", "")),
        )


@dataclass(frozen=True)
class StdIOTestCase:
    """标准输入输出模式测试用例。"""

    stdin: str
    expected_stdout: str
    name: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StdIOTestCase":
        if not isinstance(data, dict):
            raise TypeError("test case must be an object")
        if "stdin" not in data:
            raise ValueError("stdin/stdout test case requires 'stdin'")
        if "expected_stdout" not in data:
            raise ValueError("stdin/stdout test case requires 'expected_stdout'")
        if not isinstance(data["stdin"], str):
            raise TypeError("stdin/stdout test case 'stdin' must be a string")
        if not isinstance(data["expected_stdout"], str):
            raise TypeError(
                "stdin/stdout test case 'expected_stdout' must be a string"
            )
        return cls(
            stdin=data["stdin"],
            expected_stdout=data["expected_stdout"],
            name=str(data.get("name", "")),
        )


AnyTestCase: TypeAlias = FunctionTestCase | StdIOTestCase

# 向后兼容 EXP-001 及已有导入。
TestCase = FunctionTestCase


def _parse_cases(
    mode: TaskMode,
    rows: list[dict[str, Any]],
) -> tuple[AnyTestCase, ...]:
    parser = FunctionTestCase if mode == "function" else StdIOTestCase
    return tuple(parser.from_dict(row) for row in rows)


@dataclass(frozen=True)
class CodeTask:
    """同时支持函数调用与标准输入输出的可验证代码任务。"""

    task_id: str
    prompt: str
    task_mode: TaskMode
    entry_point: str | None
    starter_code: str
    public_tests: tuple[AnyTestCase, ...]
    hidden_tests: tuple[AnyTestCase, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeTask":
        if not isinstance(data, dict):
            raise TypeError("task must be an object")

        required = ("task_id", "prompt")
        missing = [key for key in required if not data.get(key)]
        if missing:
            raise ValueError(f"missing required fields: {', '.join(missing)}")

        mode = str(data.get("task_mode", "function"))
        if mode not in {"function", "stdin_stdout"}:
            raise ValueError(
                "task_mode must be either 'function' or 'stdin_stdout'"
            )

        entry_point_raw = data.get("entry_point")
        if mode == "function":
            if not entry_point_raw:
                raise ValueError("function task requires 'entry_point'")
            entry_point: str | None = str(entry_point_raw)
        else:
            entry_point = None

        public_rows = data.get("public_tests", [])
        hidden_rows = data.get("hidden_tests", [])
        if not isinstance(public_rows, list):
            raise TypeError("public_tests must be a list")
        if not isinstance(hidden_rows, list):
            raise TypeError("hidden_tests must be a list")

        public = _parse_cases(mode, public_rows)
        hidden = _parse_cases(mode, hidden_rows)

        if not public:
            raise ValueError("each task needs at least one public test")
        if not hidden:
            raise ValueError("each task needs at least one hidden test")

        return cls(
            task_id=str(data["task_id"]),
            prompt=str(data["prompt"]),
            task_mode=mode,  # type: ignore[arg-type]
            entry_point=entry_point,
            starter_code=str(data.get("starter_code", "")),
            public_tests=public,
            hidden_tests=hidden,
            metadata=dict(data.get("metadata", {})),
        )
