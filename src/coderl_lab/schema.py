from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TestCase:
    """A JSON-serializable function-level test case."""

    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    expected: Any = None
    name: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TestCase":
        if not isinstance(data, dict):
            raise TypeError("test case must be an object")
        args = data.get("args", [])
        kwargs = data.get("kwargs", {})
        if not isinstance(args, list):
            raise TypeError("test case 'args' must be a list")
        if not isinstance(kwargs, dict):
            raise TypeError("test case 'kwargs' must be an object")
        return cls(
            args=args,
            kwargs=kwargs,
            expected=data.get("expected"),
            name=str(data.get("name", "")),
        )


@dataclass(frozen=True)
class CodeTask:
    """A function-level code task with public and hidden tests."""

    task_id: str
    prompt: str
    entry_point: str
    starter_code: str
    public_tests: tuple[TestCase, ...]
    hidden_tests: tuple[TestCase, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeTask":
        required = ("task_id", "prompt", "entry_point")
        missing = [key for key in required if not data.get(key)]
        if missing:
            raise ValueError(f"missing required fields: {', '.join(missing)}")

        public = tuple(TestCase.from_dict(x) for x in data.get("public_tests", []))
        hidden = tuple(TestCase.from_dict(x) for x in data.get("hidden_tests", []))
        if not public:
            raise ValueError("each task needs at least one public test")
        if not hidden:
            raise ValueError("each task needs at least one hidden test")

        return cls(
            task_id=str(data["task_id"]),
            prompt=str(data["prompt"]),
            entry_point=str(data["entry_point"]),
            starter_code=str(data.get("starter_code", "")),
            public_tests=public,
            hidden_tests=hidden,
            metadata=dict(data.get("metadata", {})),
        )
