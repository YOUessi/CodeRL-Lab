from __future__ import annotations

import json
from pathlib import Path

import pytest

from coderl_lab.datasets.posttrain_h4 import (
    ULTRACHAT_REVISION,
    ULTRAFEEDBACK_REVISION,
    assert_disjoint,
    build_dpo_record,
    build_sft_record,
    prepare_dataset,
    select_subset,
)
from coderl_lab.train.sft import prepare_prompt_completion_rows, load_sft_rows


def sft_row(prompt: str, answer: str = "This is a valid and useful reply.") -> dict:
    return {
        "prompt_id": prompt.replace(" ", "-"),
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def dpo_row(prompt: str, answer: str = "Preferred detailed answer.", rejected: str = "Not a great response.") -> dict:
    user = {"role": "user", "content": prompt}
    return {
        "prompt_id": prompt.replace(" ", "-"),
        "chosen": [user, {"role": "assistant", "content": answer}],
        "rejected": [user, {"role": "assistant", "content": rejected}],
    }


def test_raw_instruction_sft_does_not_inject_code_task_prompt(tmp_path: Path) -> None:
    row = sft_row("How do you explain gradient accumulation?")
    fingerprint, adapted = build_sft_record(row, "train_sft")
    assert adapted["format"] == "raw"
    assert adapted["metadata"]["prompt_sha256"] == fingerprint
    assert adapted["metadata"]["source_revision"] == ULTRACHAT_REVISION

    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(adapted, ensure_ascii=False) + "\n", encoding="utf-8")
    loaded = load_sft_rows(path)
    prepared = prepare_prompt_completion_rows(loaded, seed=42)
    assert prepared[0]["prompt"].startswith("### User:\n")
    assert prepared[0]["prompt"].endswith("### Assistant:\n")
    assert "请完成下面的 Python 编程任务" not in prepared[0]["prompt"]
    assert prepared[0]["completion"] == row["messages"][1]["content"] + "\n"


def test_original_code_sft_format_still_supported() -> None:
    records = prepare_prompt_completion_rows(
        [{"task_id": "mbpp_1", "prompt": "Implement sum", "starter_code": "def add(a,b): pass", "response": "def add(a,b): return a+b"}],
        seed=42,
    )
    assert "请完成下面的 Python 编程任务" in records[0]["prompt"]
    assert "def add(a,b)" in records[0]["prompt"]


def test_dpo_shared_prefix_and_preference_contrast() -> None:
    fingerprint, record = build_dpo_record(dpo_row("Explain LoRA rank."), "train_prefs")
    assert record["metadata"]["prompt_sha256"] == fingerprint
    assert record["metadata"]["source_revision"] == ULTRAFEEDBACK_REVISION
    assert record["prompt"].endswith("### Assistant:\n")
    assert record["chosen"] != record["rejected"]


def test_dpo_rejects_incompatible_prefixes() -> None:
    row = dpo_row("Explain LoRA rank.")
    row["rejected"][0] = {"role": "user", "content": "Explain model quantization."}
    with pytest.raises(ValueError, match="prefixes disagree"):
        build_dpo_record(row, "train_prefs")


def test_dpo_rejects_identical_choices() -> None:
    with pytest.raises(ValueError, match="identical"):
        build_dpo_record(dpo_row("Compare SFT with RL.", "Identical answer.", "Identical answer."), "train_prefs")


def test_deterministic_subset_independent_of_row_order() -> None:
    rows = [sft_row(f"Explain optimizer issue number {i}") for i in range(21)]
    a = select_subset(rows, kind="sft", split="train_sft", take=5, seed=42)
    b = select_subset(list(reversed(rows)), kind="sft", split="train_sft", take=5, seed=42)
    assert [x["task_id"] for x in a.records] == [x["task_id"] for x in b.records]
    assert len(a.records) == 5
    assert a.full_scan
    assert a.candidates_scanned == 21


def test_duplicate_prompt_and_quality_filter_counted() -> None:
    rows = [
        sft_row("First long enough prompt"),
        sft_row("First long enough prompt"),
        sft_row("Second long enough prompt", "too"),
    ]
    a = select_subset(rows, kind="sft", split="train_sft", take=1, seed=42)
    assert a.rejected["duplicate_prompt"] == 1
    assert a.rejected["ValueError"] == 1


def test_smoke_scan_must_not_be_mislabeled_full_dataset() -> None:
    rows = [sft_row(f"Question number {i} in source") for i in range(16)]
    selected = select_subset(rows, kind="sft", split="train_sft", take=3, seed=42, scan_limit=5)
    assert selected.candidates_scanned == 5
    assert selected.full_scan is False


def test_split_leakage_fails_and_stops_writes(tmp_path: Path) -> None:
    base = "What is the difference between GRPO and PPO?"
    train = [sft_row(base), sft_row("First distinct train example")]
    valid = [sft_row(base), sft_row("First distinct validation example")]
    with pytest.raises(ValueError, match="prompt leakage"):
        prepare_dataset(
            kind="sft", output_dir=tmp_path, train_rows=train,
            validation_rows=valid, train_size=2, validation_size=2,
            seed=42, scan_limit=None,
        )
    assert not list(tmp_path.glob("*.jsonl"))


def test_public_manifest_contains_hash_not_raw_answers(tmp_path: Path) -> None:
    tr = [sft_row(f"Train item number {i}") for i in range(12)]
    ev = [sft_row(f"Validation item number {i}") for i in range(8)]
    manifest = prepare_dataset(
        kind="sft", output_dir=tmp_path, train_rows=tr, validation_rows=ev,
        train_size=4, validation_size=2, seed=42, scan_limit=None,
    )
    assert manifest["prompt_overlap"] == 0
    assert manifest["train"]["rows"] == 4
    assert manifest["validation"]["rows"] == 2
    assert manifest["full_scan"] is True
    assert manifest["test_or_private_examples_in_training"] is False
    assert "This is a valid" not in json.dumps(manifest)
    assert len((tmp_path / "sft_train.jsonl").read_text().splitlines()) == 4


def test_early_stop_closes_streaming_input() -> None:
    closed = []

    def rows():
        try:
            for i in range(12):
                yield sft_row(f"Data stream sample number {i}")
        finally:
            closed.append(True)

    selection = select_subset(
        rows(), kind="sft", split="train_sft",
        take=2, seed=42, scan_limit=3,
    )
    assert selection.full_scan is False
    assert closed == [True]
