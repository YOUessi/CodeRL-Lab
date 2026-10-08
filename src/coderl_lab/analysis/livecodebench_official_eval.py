from __future__ import annotations

import argparse
import base64
import io
import json
import pickle
import shutil
import subprocess
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class _RestrictedUnpickler(pickle.Unpickler):
    """Allow data-only pickle payloads, never global/class construction."""

    def find_class(self, module: str, name: str):
        raise pickle.UnpicklingError(
            f"global lookup disabled while decoding LiveCodeBench data: {module}.{name}"
        )


def _restricted_pickle_loads(data: bytes) -> Any:
    return _RestrictedUnpickler(io.BytesIO(data)).load()


def decode_test_cases(value: Any) -> list[dict[str, Any]]:
    """Decode a LiveCodeBench public/private test field.

    Upstream private tests are either plain JSON or base64(zlib(pickle(JSON-string))).
    The pickle branch is decoded with a restricted unpickler to avoid arbitrary
    class/function loading on the host.
    """
    if isinstance(value, list):
        decoded = value
    elif isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            compressed = base64.b64decode(value.encode("utf-8"))
            pickled = zlib.decompress(compressed)
            unpacked = _restricted_pickle_loads(pickled)
            if isinstance(unpacked, bytes):
                unpacked = unpacked.decode("utf-8")
            if isinstance(unpacked, str):
                decoded = json.loads(unpacked)
            elif isinstance(unpacked, list):
                decoded = unpacked
            else:
                raise TypeError(
                    "compressed LiveCodeBench tests must decode to JSON text or list"
                )
    else:
        raise TypeError("LiveCodeBench test cases must be a JSON string or list")

    if not isinstance(decoded, list) or not decoded:
        raise ValueError("LiveCodeBench test list must be non-empty")

    normalized: list[dict[str, Any]] = []
    for index, test in enumerate(decoded):
        if not isinstance(test, dict):
            raise TypeError(f"test case {index} must be an object")
        missing = [key for key in ("input", "output", "testtype") if key not in test]
        if missing:
            raise ValueError(
                f"test case {index} missing fields: {', '.join(missing)}"
            )
        testtype = str(test["testtype"])
        if testtype not in {"stdin", "functional"}:
            raise ValueError(f"unsupported LiveCodeBench testtype: {testtype}")
        normalized.append(
            {
                "input": str(test["input"]),
                "output": str(test["output"]),
                "testtype": testtype,
            }
        )
    return normalized


def build_official_sample(
    *,
    tests: Iterable[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, str]:
    test_list = list(tests)
    if not test_list:
        raise ValueError("official evaluator sample needs at least one test")

    testtypes = {str(test["testtype"]) for test in test_list}
    if len(testtypes) != 1:
        raise ValueError(
            f"mixed LiveCodeBench test types in one problem are unsupported: {testtypes}"
        )

    fn_name = metadata.get("func_name")
    if next(iter(testtypes)) == "functional" and not fn_name:
        raise ValueError("functional LiveCodeBench task is missing metadata.func_name")
    if next(iter(testtypes)) == "stdin":
        fn_name = None

    return {
        "input_output": json.dumps(
            {
                "inputs": [str(test["input"]) for test in test_list],
                "outputs": [str(test["output"]) for test in test_list],
                "fn_name": fn_name,
            },
            ensure_ascii=False,
        )
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_public_tasks(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    tasks: dict[str, dict[str, Any]] = {}
    for row in rows:
        if "private_test_cases" in row:
            raise ValueError(
                "public-only LiveCodeBench task file contains private_test_cases"
            )
        question_id = str(row["question_id"])
        if question_id in tasks:
            raise ValueError(f"duplicate LiveCodeBench question_id: {question_id}")
        tasks[question_id] = row
    return tasks


def load_private_tests_from_source(path: Path) -> dict[str, list[dict[str, Any]]]:
    rows = load_jsonl(path)
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        question_id = str(row["question_id"])
        if question_id in out:
            raise ValueError(f"duplicate upstream question_id: {question_id}")
        if "private_test_cases" not in row:
            raise ValueError(f"upstream row missing private tests: {question_id}")
        out[question_id] = decode_test_cases(row["private_test_cases"])
    return out


@dataclass(frozen=True)
class LiveCodeBenchExecutionResult:
    passed: bool
    results: tuple[Any, ...]
    metadata: dict[str, Any]


class OfficialLiveCodeBenchExecutor:
    def __init__(
        self,
        *,
        livecodebench_repo: Path,
        docker_image: str = "coderl-lab-lcb-eval",
        timeout_seconds: int = 6,
        memory_limit: str = "1g",
    ) -> None:
        if shutil.which("docker") is None:
            raise RuntimeError("docker executable not found")
        if not livecodebench_repo.exists():
            raise FileNotFoundError(
                f"pinned LiveCodeBench checkout not found: {livecodebench_repo}"
            )
        testing_util = (
            livecodebench_repo
            / "lcb_runner"
            / "evaluation"
            / "testing_util.py"
        )
        if not testing_util.exists():
            raise FileNotFoundError(f"testing_util.py missing: {testing_util}")

        inspect = subprocess.run(
            ["docker", "image", "inspect", docker_image],
            text=True,
            capture_output=True,
            check=False,
        )
        if inspect.returncode != 0:
            raise RuntimeError(
                f"Docker image '{docker_image}' is unavailable; "
                "run scripts/setup_livecodebench_official.sh first"
            )

        self.livecodebench_repo = livecodebench_repo.resolve()
        self.docker_image = docker_image
        self.timeout_seconds = int(timeout_seconds)
        self.memory_limit = memory_limit

    def run(
        self,
        *,
        code: str,
        sample: dict[str, str],
    ) -> LiveCodeBenchExecutionResult:
        in_out = json.loads(sample["input_output"])
        test_count = len(in_out["inputs"])
        if test_count <= 0:
            raise ValueError("official evaluator sample has no tests")

        payload = json.dumps(
            {
                "sample": sample,
                "code": code,
                "timeout": self.timeout_seconds,
            },
            ensure_ascii=False,
        )

        container_name = f"coderl-lcb-{uuid.uuid4().hex[:12]}"
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
            f"{self.livecodebench_repo}:/lcb:ro",
            self.docker_image,
        ]

        global_timeout = max(
            30,
            (self.timeout_seconds + 2) * test_count + 15,
        )

        try:
            proc = subprocess.run(
                command,
                input=payload,
                text=True,
                capture_output=True,
                timeout=global_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
            return LiveCodeBenchExecutionResult(
                passed=False,
                results=("global-timeout",),
                metadata={"error": "global-timeout"},
            )

        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        if not lines:
            # A generated program may exhaust the *existing frozen* 1g Docker
            # memory limit and be SIGKILLed (exit 137). A repeat of abc391_f
            # gated-low, but not its baseline, reproduced this code-specific
            # failure under the same official checker and the same private
            # cases. Resource-limit violation is an incorrect solution, not
            # a benchmark-infrastructure failure. Never raise the memory cap
            # after inspecting held-out outcomes.
            if proc.returncode == 137:
                return LiveCodeBenchExecutionResult(
                    passed=False,
                    results=("candidate-resource-limit",),
                    metadata={
                        "error": "candidate-resource-limit",
                        "exit_code": proc.returncode,
                    },
                )
            return LiveCodeBenchExecutionResult(
                passed=False,
                results=("runner-error",),
                metadata={
                    "error": proc.stderr.strip()
                    or f"container exited with {proc.returncode}"
                },
            )

        try:
            data = json.loads(lines[-1])
        except json.JSONDecodeError:
            return LiveCodeBenchExecutionResult(
                passed=False,
                results=("invalid-runner-output",),
                metadata={"error": lines[-1][:1000]},
            )

        metadata = data.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {"raw_metadata": metadata}

        results = data.get("results")
        if not isinstance(results, list):
            results = [results]

        return LiveCodeBenchExecutionResult(
            passed=bool(data.get("passed")),
            results=tuple(results),
            metadata=metadata,
        )


def evaluate_code(
    *,
    executor: OfficialLiveCodeBenchExecutor,
    task: dict[str, Any],
    code: str,
    private_tests: list[dict[str, Any]] | None = None,
) -> LiveCodeBenchExecutionResult:
    public_tests = decode_test_cases(task["public_test_cases"])
    tests = list(public_tests)
    if private_tests is not None:
        tests.extend(private_tests)
    sample = build_official_sample(
        tests=tests,
        metadata=dict(task.get("metadata", {})),
    )
    return executor.run(code=code, sample=sample)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CodeRL-Lab outputs with pinned official LiveCodeBench tests"
    )
    parser.add_argument("--public-tasks", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("public", "final"),
        default="public",
    )
    parser.add_argument("--source-jsonl", type=Path)
    parser.add_argument(
        "--livecodebench-repo",
        type=Path,
        default=Path(".external/livecodebench"),
    )
    parser.add_argument(
        "--docker-image",
        default="coderl-lab-lcb-eval",
    )
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--memory", default="1g")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "final" and args.source_jsonl is None:
        raise ValueError("--source-jsonl is required only for final private evaluation")

    tasks = load_public_tasks(args.public_tasks)
    private = (
        load_private_tests_from_source(args.source_jsonl)
        if args.mode == "final"
        else {}
    )
    predictions = load_jsonl(args.predictions)

    executor = OfficialLiveCodeBenchExecutor(
        livecodebench_repo=args.livecodebench_repo,
        docker_image=args.docker_image,
        timeout_seconds=args.timeout,
        memory_limit=args.memory,
    )

    details: list[dict[str, Any]] = []
    for row in predictions:
        question_id = str(row["question_id"])
        task = tasks[question_id]
        code = str(row.get("code", row.get("completion", "")))
        result = evaluate_code(
            executor=executor,
            task=task,
            code=code,
            private_tests=private.get(question_id) if args.mode == "final" else None,
        )
        details.append(
            {
                "question_id": question_id,
                "sample_id": int(row.get("sample_id", 0)),
                "passed": result.passed,
                "results": list(result.results),
                "metadata": result.metadata,
                "mode": args.mode,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "mode": args.mode,
        "predictions": len(details),
        "passed": sum(bool(row["passed"]) for row in details),
        "pass_rate": (
            sum(bool(row["passed"]) for row in details) / len(details)
            if details
            else 0.0
        ),
        "private_tests_accessed": args.mode == "final",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
