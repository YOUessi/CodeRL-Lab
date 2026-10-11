from __future__ import annotations

import copy
import hashlib

import pytest

from coderl_lab.analysis.train009b_termination_audit import (
    EXPECTED_MODEL_EOS,ADDITIONAL_IM_END,BUDGETS,boundary_stats,summarize,
)


def test_original_model_eos_and_qwen_turn_end_are_distinct():
    assert EXPECTED_MODEL_EOS == 151643
    assert ADDITIONAL_IM_END == 151645
    assert EXPECTED_MODEL_EOS != ADDITIONAL_IM_END


def test_derived_cap_prefix_stops_naturally_only_at_true_eos():
    ids=[101]*175 + [EXPECTED_MODEL_EOS]
    v=boundary_stats(ids)
    assert v["generated_tokens"]==176
    assert v["default_eos_position"]==176
    assert v["budgets"]["96"]["default_eos_terminated"] is False
    assert v["budgets"]["96"]["would_hit_cap"] is True
    assert v["budgets"]["256"]["default_eos_terminated"] is True
    assert v["budgets"]["256"]["completion_tokens"]==176
    assert v["budgets"]["512"]["completion_tokens"]==176


def test_im_end_is_not_silently_promoted_to_generation_eos():
    ids=[5]*45+[ADDITIONAL_IM_END]+[5]*466
    assert len(ids)==512
    v=boundary_stats(ids)
    assert v["im_end_seen"] is True
    assert v["default_eos_seen"] is False
    assert v["budgets"]["96"]["im_end_seen_by_cap"] is True
    assert v["budgets"]["256"]["would_hit_cap"] is True
    assert v["budgets"]["512"]["would_hit_cap"] is True


def test_one_default_eos_must_be_true_terminal():
    with pytest.raises(ValueError,match="after forced EOS"):
        boundary_stats([151643,5])
    with pytest.raises(ValueError,match="empty"):
        boundary_stats([])
    with pytest.raises(ValueError,match="invalid"):
        boundary_stats([5],budgets=(96,96))


def frozen_32():
    rows=[]
    for cohort in ("train_smoke","sft_heldout"):
        for i in range(16):
            ids=[42]*95+[EXPECTED_MODEL_EOS] if i==0 else [7]*512
            rows.append({
                "split":cohort,
                "prompt_sha256":hashlib.sha256(f"{cohort}-{i}".encode()).hexdigest(),
                "trace":boundary_stats(ids),
            })
    return rows


def test_16_by_16_terminal_aggregation_is_complete_and_deterministic():
    s=summarize(frozen_32())
    assert s["source_prompts"] == {"train_smoke":16,"sft_heldout":16}
    assert s["metrics"]["train_smoke"]["96"]["default_eos_terminated"]==1
    assert s["metrics"]["train_smoke"]["512"]["hit_cap"]==15
    assert s["metrics"]["sft_heldout"]["256"]["default_eos_terminated"]==1
    assert s["no_training"] is True
    assert s["no_hidden_test_use"] is True


def test_partial_heldout_or_duplicate_prompt_fails():
    rows=frozen_32()
    with pytest.raises(ValueError,match="full 16"):
        summarize(rows[:30])
    dup=copy.deepcopy(rows)
    dup[-1]["prompt_sha256"]=dup[-2]["prompt_sha256"]
    with pytest.raises(ValueError,match="duplicate"):
        summarize(dup)
