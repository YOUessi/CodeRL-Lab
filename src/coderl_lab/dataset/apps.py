from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from coderl_lab.schema import CodeTask


DEFAULT_APPS_REPOSITORY = "codeparrot/apps"
DEFAULT_APPS_REVISION = "21e74ddf8de1a21436da12e3e653065c5213e9d1"


class AppsAdapterError(ValueError):
    """一条 APPS 数据无法安全映射到当前任务协议。"""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail or reason


@dataclass(frozen=True)
class AppsAdapterConfig:
    seed: int = 42
    min_tests: int = 4


def _parse_json_field(value: Any, field_name: str) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise AppsAdapterError(
                f"invalid_{field_name}_json",
                f"{field_name} is not valid JSON",
            ) from exc
    return value


def _parse_reference_solutions(value: Any) -> list[str]:
    parsed = _parse_json_field(value, "solutions")
    if not isinstance(parsed, list):
        raise AppsAdapterError(
            "invalid_solutions_type",
            "solutions must decode to a list",
        )
    solutions = [item for item in parsed if isinstance(item, str) and item.strip()]
    if not solutions:
        raise AppsAdapterError(
            "missing_reference_solutions",
            "no usable Python reference solution",
        )
    return solutions


def _public_test_count(total_tests: int) -> int:
    if total_tests < 4:
        raise AppsAdapterError(
            "too_few_tests",
            f"requires >= 4 tests, got {total_tests}",
        )
    if total_tests <= 5:
        return 1
    if total_tests <= 9:
        return 2
    return max(2, int(total_tests * 0.20))


def _deterministic_public_indices(
    *,
    task_id: str,
    total_tests: int,
    public_count: int,
    seed: int,
) -> set[int]:
    digest = hashlib.sha256(f"{seed}:{task_id}".encode("utf-8")).digest()
    local_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    rng = random.Random(local_seed)
    indices = list(range(total_tests))
    rng.shuffle(indices)
    return set(indices[:public_count])


def _stdio_text(value: Any, field_name: str) -> str:
    if isinstance(value, str):
        return value

    # 某些数据导出会把单一文本额外包一层列表；只处理这个无歧义情况。
    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], str)
    ):
        return value[0]

    raise AppsAdapterError(
        f"unsupported_{field_name}_format",
        f"{field_name} must be a string or a one-string list",
    )


def convert_apps_row(
    row: dict[str, Any],
    *,
    split: str,
    source_revision: str = DEFAULT_APPS_REVISION,
    config: AppsAdapterConfig = AppsAdapterConfig(),
) -> dict[str, Any]:
    """把一条 APPS 原始记录转换为 CodeRL-Lab JSON 任务。"""

    problem_id = row.get("problem_id", row.get("id"))
    question = row.get("question")
    if problem_id is None:
        raise AppsAdapterError("missing_problem_id")
    if not isinstance(question, str) or not question.strip():
        raise AppsAdapterError("missing_question")

    task_id = f"apps/{split}/{problem_id}"

    io = _parse_json_field(row.get("input_output"), "input_output")
    if not isinstance(io, dict):
        raise AppsAdapterError(
            "invalid_input_output_type",
            "input_output must decode to an object",
        )

    inputs = io.get("inputs")
    outputs = io.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        raise AppsAdapterError(
            "missing_inputs_or_outputs",
            "input_output requires list inputs and outputs",
        )
    if len(inputs) != len(outputs):
        raise AppsAdapterError(
            "input_output_length_mismatch",
            f"inputs={len(inputs)} outputs={len(outputs)}",
        )
    if len(inputs) < config.min_tests:
        raise AppsAdapterError(
            "too_few_tests",
            f"requires >= {config.min_tests} tests, got {len(inputs)}",
        )

    fn_name = io.get("fn_name")
    function_mode = isinstance(fn_name, str) and bool(fn_name.strip())
    task_mode = "function" if function_mode else "stdin_stdout"

    tests: list[dict[str, Any]] = []
    for index, (raw_input, raw_output) in enumerate(zip(inputs, outputs)):
        if function_mode:
            args = raw_input if isinstance(raw_input, list) else [raw_input]
            tests.append(
                {
                    "name": f"apps_{index}",
                    "args": args,
                    "kwargs": {},
                    "expected": raw_output,
                }
            )
        else:
            tests.append(
                {
                    "name": f"apps_{index}",
                    "stdin": _stdio_text(raw_input, "stdin"),
                    "expected_stdout": _stdio_text(
                        raw_output,
                        "expected_stdout",
                    ),
                }
            )

    public_count = _public_test_count(len(tests))
    public_indices = _deterministic_public_indices(
        task_id=task_id,
        total_tests=len(tests),
        public_count=public_count,
        seed=config.seed,
    )
    public_tests = [
        test for index, test in enumerate(tests) if index in public_indices
    ]
    hidden_tests = [
        test for index, test in enumerate(tests) if index not in public_indices
    ]

    solutions = _parse_reference_solutions(row.get("solutions"))

    task = {
        "task_id": task_id,
        "prompt": question.strip(),
        "task_mode": task_mode,
        "entry_point": fn_name.strip() if function_mode else None,
        "starter_code": str(row.get("starter_code") or ""),
        "public_tests": public_tests,
        "hidden_tests": hidden_tests,
        "reference_solutions": solutions,
        "metadata": {
            "source": "APPS",
            "source_dataset": DEFAULT_APPS_REPOSITORY,
            "source_revision": source_revision,
            "source_split": split,
            "source_problem_id": str(problem_id),
            "difficulty": row.get("difficulty"),
            "url": row.get("url"),
            "adapter_version": 1,
            "split_seed": config.seed,
            "original_test_count": len(tests),
            "public_test_indices": sorted(public_indices),
            "hidden_test_indices": sorted(
                set(range(len(tests))) - public_indices
            ),
        },
    }

    # 用核心协议做最后一次强校验，防止适配器产生自身无法读取的数据。
    CodeTask.from_dict(task)
    return task


def _iter_apps_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid APPS JSONL at line {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise RuntimeError(
                    f"APPS line {line_number} is not a JSON object"
                )
            yield row


def _download_apps_jsonl(
    *,
    repository: str,
    revision: str,
    split: str,
) -> Path:
    if split not in {"train", "test"}:
        raise ValueError("APPS split must be 'train' or 'test'")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "dataset dependencies are missing; install with "
            "pip install -e '.[data]'"
        ) from exc

    path = hf_hub_download(
        repo_id=repository,
        filename=f"{split}.jsonl",
        repo_type="dataset",
        revision=revision,
    )
    return Path(path)


def convert_apps_dataset(
    *,
    split: str,
    output_path: Path,
    report_path: Path,
    repository: str = DEFAULT_APPS_REPOSITORY,
    revision: str = DEFAULT_APPS_REVISION,
    limit: int | None = None,
    difficulty: str | None = None,
    config: AppsAdapterConfig = AppsAdapterConfig(),
) -> dict[str, Any]:
    raw_path = _download_apps_jsonl(
        repository=repository,
        revision=revision,
        split=split,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    skipped: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    difficulties: Counter[str] = Counter()
    test_counts: Counter[str] = Counter()
    seen = 0
    converted = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for row in _iter_apps_jsonl(raw_path):
            seen += 1

            row_difficulty = row.get("difficulty")
            if difficulty and row_difficulty != difficulty:
                skipped["difficulty_filter"] += 1
                continue

            try:
                task = convert_apps_row(
                    row,
                    split=split,
                    source_revision=revision,
                    config=config,
                )
            except AppsAdapterError as exc:
                skipped[exc.reason] += 1
                continue

            handle.write(json.dumps(task, ensure_ascii=False) + "\n")
            converted += 1
            modes[task["task_mode"]] += 1
            difficulties[str(task["metadata"].get("difficulty"))] += 1
            count = int(task["metadata"]["original_test_count"])
            if count < 6:
                bucket = "4-5"
            elif count < 10:
                bucket = "6-9"
            elif count < 20:
                bucket = "10-19"
            else:
                bucket = "20+"
            test_counts[bucket] += 1

            if limit is not None and converted >= limit:
                break

    report = {
        "source_dataset": repository,
        "source_revision": revision,
        "source_split": split,
        "seed": config.seed,
        "min_tests": config.min_tests,
        "requested_limit": limit,
        "difficulty_filter": difficulty,
        "rows_seen": seen,
        "tasks_converted": converted,
        "tasks_skipped": sum(skipped.values()),
        "skip_reasons": dict(sorted(skipped.items())),
        "task_modes": dict(sorted(modes.items())),
        "difficulties": dict(sorted(difficulties.items())),
        "original_test_count_buckets": dict(sorted(test_counts.items())),
    }

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert APPS to CodeRL-Lab")
    parser.add_argument("--repository", default=DEFAULT_APPS_REPOSITORY)
    parser.add_argument("--revision", default=DEFAULT_APPS_REVISION)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--difficulty")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-tests", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_apps_dataset(
        repository=args.repository,
        revision=args.revision,
        split=args.split,
        output_path=args.output,
        report_path=args.report,
        limit=args.limit,
        difficulty=args.difficulty,
        config=AppsAdapterConfig(
            seed=args.seed,
            min_tests=args.min_tests,
        ),
    )


if __name__ == "__main__":
    main()
