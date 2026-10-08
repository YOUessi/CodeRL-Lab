from __future__ import annotations

import base64
import json
import pickle
import zlib
from pathlib import Path

from coderl_lab.analysis.livecodebench_official_eval import (
    build_official_sample,
    decode_test_cases,
)
from coderl_lab.datasets.livecodebench import (
    LiveCodeBenchPublicTask,
    build_generic_base_prompt,
    prepare_public_view,
)


def _row(*, private_value: str = "PRIVATE_SENTINEL") -> dict:
    return {
        "question_title": "Example",
        "question_content": "Return x + 1.",
        "platform": "leetcode",
        "question_id": "q1",
        "contest_id": "c1",
        "contest_date": "2025-01-01T00:00:00",
        "starter_code": "class Solution:\n    def addOne(self, x: int) -> int:\n        ",
        "difficulty": "easy",
        "public_test_cases": json.dumps(
            [
                {
                    "input": "[1]",
                    "output": "2",
                    "testtype": "functional",
                }
            ]
        ),
        "private_test_cases": private_value,
        "metadata": json.dumps({"func_name": "addOne"}),
    }


def test_public_view_never_exports_private_tests(tmp_path: Path) -> None:
    source = tmp_path / "test6.jsonl"
    source.write_text(
        json.dumps(_row(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "public"
    manifest = prepare_public_view(
        source_path=source,
        output_dir=out,
        source_revision="fixture",
    )

    text = (out / "public_tasks.jsonl").read_text(encoding="utf-8")
    assert "PRIVATE_SENTINEL" not in text
    assert "private_test_cases" not in text
    assert manifest["private_tests_exported"] is False
    assert manifest["private_tests_decoded"] is False


def test_private_decoder_supports_plain_json() -> None:
    tests = [
        {"input": "1\n", "output": "2\n", "testtype": "stdin"}
    ]
    assert decode_test_cases(json.dumps(tests)) == tests


def test_private_decoder_supports_compressed_pickle_json() -> None:
    tests = [
        {"input": "[1]", "output": "2", "testtype": "functional"}
    ]
    packed = pickle.dumps(json.dumps(tests))
    encoded = base64.b64encode(zlib.compress(packed)).decode("utf-8")
    assert decode_test_cases(encoded) == tests


def test_official_sample_uses_func_name() -> None:
    tests = [
        {"input": "[1]", "output": "2", "testtype": "functional"}
    ]
    sample = build_official_sample(
        tests=tests,
        metadata={"func_name": "addOne"},
    )
    payload = json.loads(sample["input_output"])
    assert payload["fn_name"] == "addOne"
    assert payload["inputs"] == ["[1]"]
    assert payload["outputs"] == ["2"]


def test_generic_base_prompt_uses_official_first_example(tmp_path: Path) -> None:
    examples = (
        tmp_path
        / "lcb_runner"
        / "prompts"
        / "few_shot_examples"
        / "generation"
    )
    examples.mkdir(parents=True)
    (examples / "func.json").write_text(
        json.dumps(
            [
                {
                    "question": "OFFICIAL_EXAMPLE_QUESTION",
                    "sample_code": "OFFICIAL_STARTER",
                    "answer": "OFFICIAL_ANSWER",
                }
            ]
        ),
        encoding="utf-8",
    )
    (examples / "stdin.json").write_text(
        json.dumps(
            [
                {
                    "question": "STDIN_EXAMPLE",
                    "answer": "STDIN_ANSWER",
                }
            ]
        ),
        encoding="utf-8",
    )

    task = LiveCodeBenchPublicTask.from_upstream_row(_row())
    prompt = build_generic_base_prompt(task, livecodebench_repo=tmp_path)

    assert "OFFICIAL_EXAMPLE_QUESTION" in prompt
    assert "OFFICIAL_STARTER" in prompt
    assert "OFFICIAL_ANSWER" in prompt
    assert "Return x + 1." in prompt
    assert task.starter_code in prompt


def test_v5_public_only_stream_preserves_v6_default(tmp_path: Path) -> None:
    source = tmp_path / "test5.jsonl"
    first = _row(private_value="NEVER_EXPORT_V5_PRIVATE")
    second = _row(private_value="SENTINEL_TWO")
    second["question_id"] = "q2"
    source.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n",
        encoding="utf-8",
    )
    v5 = prepare_public_view(
        source_path=source,
        output_dir=tmp_path / "v5",
        source_revision="frozen",
        fine_grained_version="v5",
    )
    assert v5["tasks"] == 2
    assert v5["source_filename"] == "test5.jsonl"
    assert v5["fine_grained_version"] == "v5"
    assert not v5["private_tests_decoded"]
    assert "NEVER_EXPORT" not in (tmp_path / "v5/public_tasks.jsonl").read_text()

    v6 = prepare_public_view(
        source_path=source, output_dir=tmp_path / "v6",
        source_revision="frozen",
    )
    assert v6["source_filename"] == "test6.jsonl"
    assert v6["fine_grained_version"] == "v6"
    assert v6["public_view_sha256"] == v5["public_view_sha256"]


def test_public_only_source_duplicate_question_fails(tmp_path: Path) -> None:
    source = tmp_path / "dup.jsonl"
    source.write_text(
        json.dumps(_row()) + "\n" + json.dumps(_row()) + "\n",
        encoding="utf-8",
    )
    import pytest
    with pytest.raises(ValueError, match="duplicate LiveCodeBench question_id"):
        prepare_public_view(
            source_path=source, output_dir=tmp_path / "out",
            source_revision="fixture", fine_grained_version="v5",
        )
