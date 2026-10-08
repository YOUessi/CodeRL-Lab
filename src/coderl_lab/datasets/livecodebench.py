from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATASET_REPO = "livecodebench/code_generation_lite"
V6_FILENAME = "test6.jsonl"


@dataclass(frozen=True)
class LiveCodeBenchPublicTask:
    question_id: str
    question_title: str
    question_content: str
    platform: str
    contest_id: str
    contest_date: str
    starter_code: str
    difficulty: str
    public_test_cases: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]

    @classmethod
    def from_upstream_row(cls, row: dict[str, Any]) -> "LiveCodeBenchPublicTask":
        public_raw = row.get("public_test_cases")
        metadata_raw = row.get("metadata")

        if not isinstance(public_raw, str):
            raise TypeError("public_test_cases must be a JSON string")
        if not isinstance(metadata_raw, str):
            raise TypeError("metadata must be a JSON string")

        public = json.loads(public_raw)
        metadata = json.loads(metadata_raw)

        if not isinstance(public, list) or not public:
            raise ValueError("public_test_cases must be a non-empty JSON list")
        if not isinstance(metadata, dict):
            raise ValueError("metadata must decode to an object")

        normalized_tests: list[dict[str, Any]] = []
        for index, test in enumerate(public):
            if not isinstance(test, dict):
                raise TypeError(f"public test {index} is not an object")
            missing = [key for key in ("input", "output", "testtype") if key not in test]
            if missing:
                raise ValueError(
                    f"public test {index} missing fields: {', '.join(missing)}"
                )
            testtype = str(test["testtype"])
            if testtype not in {"stdin", "functional"}:
                raise ValueError(f"unsupported LiveCodeBench testtype: {testtype}")
            normalized_tests.append(
                {
                    "input": str(test["input"]),
                    "output": str(test["output"]),
                    "testtype": testtype,
                }
            )

        required = (
            "question_id",
            "question_title",
            "question_content",
            "platform",
            "contest_id",
            "contest_date",
            "starter_code",
            "difficulty",
        )
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"upstream row missing fields: {', '.join(missing)}")

        return cls(
            question_id=str(row["question_id"]),
            question_title=str(row["question_title"]),
            question_content=str(row["question_content"]),
            platform=str(row["platform"]),
            contest_id=str(row["contest_id"]),
            contest_date=str(row["contest_date"]),
            starter_code=str(row["starter_code"]),
            difficulty=str(row["difficulty"]),
            public_test_cases=tuple(normalized_tests),
            metadata=metadata,
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question_title": self.question_title,
            "question_content": self.question_content,
            "platform": self.platform,
            "contest_id": self.contest_id,
            "contest_date": self.contest_date,
            "starter_code": self.starter_code,
            "difficulty": self.difficulty,
            "public_test_cases": list(self.public_test_cases),
            "metadata": self.metadata,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_v6_source() -> tuple[Path, str]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "LiveCodeBench data download requires huggingface_hub; "
            "install the project data/model extras before preparing v6"
        ) from exc

    api = HfApi()
    info = api.dataset_info(DATASET_REPO, revision="main")
    revision = str(info.sha)
    path = Path(
        hf_hub_download(
            repo_id=DATASET_REPO,
            filename=V6_FILENAME,
            repo_type="dataset",
            revision=revision,
        )
    )
    return path, revision


def load_upstream_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("LiveCodeBench source is empty")
    return rows


def prepare_public_view(
    *,
    source_path: Path,
    output_dir: Path,
    source_revision: str,
    limit: int | None = None,
) -> dict[str, Any]:
    rows = load_upstream_rows(source_path)
    if limit is not None:
        rows = rows[:limit]

    tasks = [LiveCodeBenchPublicTask.from_upstream_row(row) for row in rows]

    output_dir.mkdir(parents=True, exist_ok=True)
    public_path = output_dir / "public_tasks.jsonl"
    with public_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            record = task.to_public_dict()
            serialized = json.dumps(record, ensure_ascii=False)
            if "private_test_cases" in serialized:
                raise AssertionError("private_test_cases leaked into public view")
            handle.write(serialized + "\n")

    platforms = Counter(task.platform for task in tasks)
    difficulties = Counter(task.difficulty for task in tasks)
    testtypes = Counter(
        str(test["testtype"])
        for task in tasks
        for test in task.public_test_cases
    )
    public_counts = [len(task.public_test_cases) for task in tasks]
    dates = sorted(task.contest_date for task in tasks)

    manifest = {
        "dataset_repo": DATASET_REPO,
        "fine_grained_version": "v6",
        "source_filename": V6_FILENAME,
        "source_revision": source_revision,
        "source_sha256": sha256_file(source_path),
        "public_view_path": str(public_path),
        "public_view_sha256": sha256_file(public_path),
        "tasks": len(tasks),
        "platform_counts": dict(sorted(platforms.items())),
        "difficulty_counts": dict(sorted(difficulties.items())),
        "testtype_counts": dict(sorted(testtypes.items())),
        "public_tests_total": sum(public_counts),
        "public_tests_min": min(public_counts),
        "public_tests_max": max(public_counts),
        "contest_date_min": dates[0],
        "contest_date_max": dates[-1],
        "private_tests_exported": False,
        "private_tests_decoded": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _example_prompt(example: dict[str, Any], *, has_starter: bool) -> str:
    prompt = "### Question\n" + str(example["question"]) + "\n\n"
    if has_starter:
        prompt += "### Starter Code\n" + str(example["sample_code"]) + "\n\n"
    prompt += "### Answer\n\n" + str(example.get("answer", ""))
    if example.get("answer"):
        prompt += "\n\n"
    return prompt


def build_generic_base_prompt(
    task: LiveCodeBenchPublicTask,
    *,
    livecodebench_repo: Path,
) -> str:
    filename = "func.json" if task.starter_code else "stdin.json"
    examples_path = (
        livecodebench_repo
        / "lcb_runner"
        / "prompts"
        / "few_shot_examples"
        / "generation"
        / filename
    )
    examples = json.loads(examples_path.read_text(encoding="utf-8"))
    if not isinstance(examples, list) or not examples:
        raise ValueError(f"official LiveCodeBench example file is empty: {examples_path}")

    has_starter = bool(task.starter_code)
    prompt = _example_prompt(examples[0], has_starter=has_starter)
    prompt += _example_prompt(
        {
            "question": task.question_content,
            "sample_code": task.starter_code,
            "answer": "",
        },
        has_starter=has_starter,
    )
    return prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the public-only LiveCodeBench v6 view"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source is None:
        source_path, source_revision = download_v6_source()
    else:
        source_path = args.source
        source_revision = args.source_revision or "local-file"

    manifest = prepare_public_view(
        source_path=source_path,
        output_dir=args.output_dir,
        source_revision=source_revision,
        limit=args.limit,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
