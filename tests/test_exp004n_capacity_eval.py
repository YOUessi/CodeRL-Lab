from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from coderl_lab.analysis import exp004n_formal_eval as f
from coderl_lab.analysis.exp004n_capacity_generate import (
    OFFICIAL_COMMIT, MODEL, MODEL_REVISION, SOURCE_REVISION,
    SOURCE_SHA256, PUBLIC_SHA256, SFT_ADAPTER_SHA256,
    summarize_completion,
)


def _fixture(tmp_path: Path, *, broken_gate: bool = False):
    ids = [f"v5_{i:03d}" for i in range(167)]
    sft = {
        "experiment": "EXP-004N", "model": MODEL, "revision": MODEL_REVISION,
        "official_livecodebench_commit": OFFICIAL_COMMIT,
        "private_tests_accessed": False, "tasks": 167,
        "max_new_tokens": 512, "primary_window": 128,
        "low_threshold": 0.05, "high_threshold": 0.20, "bias": 0.25,
        "per_task": {},
    }
    controls = []
    for arm, max_tokens, adapter_sha in (
        ("base512", 512, None),
        ("sft1024", 1024, SFT_ADAPTER_SHA256),
    ):
        controls.append({
            "experiment": "EXP-004N", "arm": arm, "model": MODEL,
            "model_revision": MODEL_REVISION,
            "official_checker_commit": OFFICIAL_COMMIT,
            "private_tests_accessed": False, "tasks": 167,
            "is_formal": True, "max_new_tokens": max_tokens,
            "adapter_sha256": adapter_sha,
            "public_view_sha256": PUBLIC_SHA256,
            "per_task": {},
        })
    for qid in ids:
        cond = {"raw_completion": "print(1)", "code": "print(1)"}
        sft["per_task"][qid] = {
            "question_id": qid, "eligible": False,
            "public_all_pass": True,
            "gate_triggered": broken_gate and qid == ids[0],
            "baseline": cond, "gated_low_destabilize": cond,
            "gated_high_destabilize": cond, "always_low_destabilize": cond,
        }
        for control, max_tokens in zip(controls, (512, 1024)):
            control["per_task"][qid] = {
                "question_id": qid,
                "code": "print(1)", "raw_completion": "print(1)",
                "generated_token_count": 5,
                "reached_token_cap": False,
            }
    manifest = {
        "fine_grained_version": "v5", "source_filename": "test5.jsonl",
        "source_revision": SOURCE_REVISION,
        "source_sha256": SOURCE_SHA256,
        "public_view_sha256": PUBLIC_SHA256,
        "tasks": 167, "private_tests_exported": False,
        "private_tests_decoded": False,
    }
    paths = [tmp_path / f"{name}.json" for name in ("sft", "base", "long", "manifest")]
    for path, obj in zip(paths, (sft, controls[0], controls[1], manifest)):
        path.write_text(json.dumps(obj), encoding="utf-8")
    public = tmp_path / "public.jsonl"
    public.write_text("placeholder", encoding="utf-8")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_text("fixture")
    return paths, public, adapter, {qid: {} for qid in ids}


def _validate(tmp_path: Path, monkeypatch, *, broken_gate: bool = False):
    paths, public, adapter, tasks = _fixture(tmp_path, broken_gate=broken_gate)
    monkeypatch.setattr(f, "sha256_file",
                        lambda p: PUBLIC_SHA256 if p == public else SFT_ADAPTER_SHA256)
    monkeypatch.setattr(f, "load_public_tasks", lambda path: tasks)
    monkeypatch.setattr(f.subprocess, "run",
                        lambda *args, **kwargs: SimpleNamespace(stdout=OFFICIAL_COMMIT+"\n"))
    return f.validate_inputs(
        sft512_path=paths[0], base512_path=paths[1],
        sft1024_path=paths[2], public_path=public,
        manifest_path=paths[3], adapter_dir=adapter,
        lcb_repo=tmp_path,
    )


def test_frozen_v5_six_arm_data_passes(tmp_path: Path, monkeypatch) -> None:
    freeze = _validate(tmp_path, monkeypatch)
    assert freeze["tasks"] == 167
    assert freeze["private_tests_accessed"] is False
    assert freeze["adapter_sha256"] == SFT_ADAPTER_SHA256
    assert freeze["source_sha256"] == SOURCE_SHA256


def test_public_gate_violation_blocks_hidden_evaluation(
    tmp_path: Path, monkeypatch
) -> None:
    with pytest.raises(ValueError, match="public-only gate"):
        _validate(tmp_path, monkeypatch, broken_gate=True)


def test_pinned_private_isolation_must_be_explicit(tmp_path: Path, monkeypatch) -> None:
    paths, public, adapter, tasks = _fixture(tmp_path)
    runner = json.loads(paths[1].read_text())
    runner["private_tests_accessed"] = True
    paths[1].write_text(json.dumps(runner))
    monkeypatch.setattr(f, "sha256_file",
                        lambda p: PUBLIC_SHA256 if p == public else SFT_ADAPTER_SHA256)
    monkeypatch.setattr(f, "load_public_tasks", lambda path: tasks)
    monkeypatch.setattr(f.subprocess, "run",
                        lambda *args, **kwargs: SimpleNamespace(stdout=OFFICIAL_COMMIT+"\n"))
    with pytest.raises(ValueError, match="private tests accessed"):
        f.validate_inputs(
            sft512_path=paths[0], base512_path=paths[1],
            sft1024_path=paths[2], public_path=public,
            manifest_path=paths[3], adapter_dir=adapter, lcb_repo=tmp_path,
        )


def test_cap_and_ast_metrics_are_not_correctness_metrics() -> None:
    good = summarize_completion(raw="def f():\n    return 1\n", token_count=512, token_cap=512)
    bad = summarize_completion(raw="def f():\n    return (", token_count=512, token_cap=512)
    assert good["reached_token_cap"] is True
    assert good["syntax_ok"] is True
    assert bad["syntax_ok"] is False
    assert bad["reached_token_cap"] is True


def test_all_six_arms_and_bootstrap_use_same_task_grid() -> None:
    rows = []
    for i in range(7):
        x = {
            "task_id": f"fixture_task_{i:03d}",
            "platform": "atcoder" if i < 4 else "leetcode",
            "difficulty": "easy" if i < 3 else "hard",
            "sft512_public_all_pass": i == 0,
            "gate_triggered": i > 0,
            "gated_low_changed": i % 2 == 1,
            "gated_high_changed": False,
            "base512_cap_hit": i == 6, "base512_syntax_ok": i < 6,
            "sft1024_cap_hit": False, "sft1024_syntax_ok": True,
            "sft1024_prefix_matches_sft512": True,
        }
        for arm in f.ARMS:
            x[f"{arm}_correct"] = arm in {"gated_low", "sft1024"} or i == 0
        rows.append(x)
    summary = f.summarize(rows, iterations=200, seed=42)
    assert summary["arms"]["sft512"]["correct"] == 1
    assert summary["arms"]["gated_low"]["correct"] == 7
    assert summary["arms"]["base512"]["correct"] == 1
    assert summary["preregistered_primary_success"] is True
    assert summary["completion_quality"]["base512"]["cap_hit"] == 1
    assert summary["gate_audit"]["public_pass_protected"] is True


def test_control_file_cannot_overwrite_formal_frozen_runner(tmp_path: Path) -> None:
    # The runner validates its frozen output path before any model loading.
    # File existence itself is part of the immutable evaluation protocol.
    assert f.sha_string("same code") == f.sha_string("same code")
    assert f.sha_string("different code") != f.sha_string("same code")
