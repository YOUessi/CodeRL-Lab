import json

import pytest

from coderl_lab.datasets.exp7c_humanevalplus import (
    ALLOWED_FIELDS,
    public_rows_from_source,
    write_public,
)
from coderl_lab.analysis.exp7c_matched_generate import (
    strip_outer_code_fence, structural_check, seed_for,
    prompt_messages, read_checkpoint, sha_text,
)


def sample(task="HumanEval/0"):
    return {
        "task_id": task,
        "prompt": "def add_one(x):\n    \"\"\"Return x+1.\"\"\"\n    pass",
        "entry_point": "add_one",
        "canonical_solution": "return x+1",
        "test": "def check(candidate): assert candidate(3)==4",
        "base_input": ["3"],
        "plus_input": ["5"],
    }


def test_whitelist_filters_all_solutions_tests_and_preserves_signature(tmp_path):
    public = public_rows_from_source([sample()], expected_count=1)
    assert set(public[0]) == ALLOWED_FIELDS
    assert not any(k in public[0] for k in ("test", "canonical_solution", "base_input", "plus_input"))
    manifest = write_public(public, tmp_path)
    assert manifest["tasks"] == 1
    assert manifest["reference_solutions_exported"] is False
    assert json.loads((tmp_path / "public_tasks.jsonl").read_text()) == public[0]
    with pytest.raises(FileExistsError):
        write_public(public, tmp_path)


def test_bad_dataset_rejected():
    with pytest.raises(ValueError, match="expected"):
        public_rows_from_source([], expected_count=1)
    with pytest.raises(ValueError, match="duplicated"):
        public_rows_from_source([sample(), sample()], expected_count=2)
    wrong = sample()
    wrong["entry_point"] = "not_present"
    with pytest.raises(ValueError, match="entry point"):
        public_rows_from_source([wrong], expected_count=1)


def test_code_normalization_frozen_before_scores():
    ticks = chr(96) * 3
    raw = ticks + "python\ndef add_one(x):\n    return x + 1\n" + ticks + "\n\nExplanation..."
    code, reason = strip_outer_code_fence(raw)
    assert reason == "first_closed_python_fence"
    assert code == "def add_one(x):\n    return x + 1"
    assert structural_check(code, "add_one") == (True, True)
    incomplete = ticks + "python\ndef add_one(x):\n    return x + 1"
    result, typ = strip_outer_code_fence(incomplete)
    assert result == incomplete and typ == "unclosed_fence_preserved"
    assert structural_check(result, "add_one") == (False, False)


def test_same_seed_and_public_prompt_no_lookahead():
    row = {k: v for k, v in sample().items() if k in ALLOWED_FIELDS}
    assert seed_for(row["task_id"], 0) == seed_for(row["task_id"], 0)
    assert seed_for(row["task_id"], 0) != seed_for(row["task_id"], 1)
    messages = prompt_messages(row)
    text = json.dumps(messages)
    assert "canonical_solution" not in text and "test" not in text
    with pytest.raises(ValueError, match="forbidden"):
        prompt_messages(sample())


def test_resume_does_not_accept_mutated_code_or_samples(tmp_path):
    row = {k: v for k, v in sample().items() if k in ALLOWED_FIELDS}
    raw = "def add_one(x):\n    return x+1"
    checkpoint = tmp_path / "checkpoint.jsonl"
    x = {
        "task_id": row["task_id"], "arm": "base", "sample_id": 0,
        "raw": raw, "raw_sha256": sha_text(raw),
        "code": raw, "code_sha256": sha_text(raw),
        "extraction": "unfenced_unmodified", "seed": seed_for(row["task_id"], 0),
    }
    checkpoint.write_text(json.dumps(x)+"\n")
    assert len(read_checkpoint(checkpoint, [row], 1)) == 1
    x["code"] = "def add_one(x): return 0"
    checkpoint.write_text(json.dumps(x)+"\n")
    with pytest.raises(ValueError, match="checkpoint"):
        read_checkpoint(checkpoint, [row], 1)
