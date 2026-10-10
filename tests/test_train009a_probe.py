from __future__ import annotations

import math

import pytest

from coderl_lab.analysis.train009a_probe import summarize_probe


def _record(i, ref, ppo, ref_score=0.2, ppo_score=0.3):
    return {
        "prompt_sha256":f"prompt-{i}",
        "reference_answer_sha256":ref,
        "ppo_answer_sha256":ppo,
        "reference_tokens":32,
        "ppo_tokens":31,
        "reference_score":ref_score,
        "ppo_score":ppo_score,
    }


def test_preselected_four_probe_scores_are_descriptive_no_labels():
    rows=[_record(0,"A","B"),_record(1,"C","C"),
          _record(2,"D","D"),_record(3,"E","E")]
    o=summarize_probe(rows)
    assert o["experiment"]=="TRAIN-009A-probe"
    assert o["probe_count"]==4
    assert o["changed_greedy_outputs"]==1
    assert o["mean_frozen_RM_score_ppo"]-o["mean_frozen_RM_score_reference"]==pytest.approx(0.1)
    assert o["no_human_labels_used"] is True
    assert o["no_optimizer_updates"] is True
    assert any("NOT evidence of human" in s for s in o["limits"])


def test_probe_does_not_select_based_on_reward_or_allow_missing_data():
    rows=[_record(i,"a","b") for i in range(4)]
    with pytest.raises(ValueError,match="four"):
        summarize_probe(rows[:3])
    duplicate=rows.copy()
    duplicate[-1]=_record(0,"a","b")
    with pytest.raises(ValueError,match="distinct"):
        summarize_probe(duplicate)
    missing=rows.copy()
    missing[-1]={k:v for k,v in missing[-1].items() if k!="ppo_score"}
    with pytest.raises(ValueError,match="incomplete"):
        summarize_probe(missing)
    nan=rows.copy()
    nan[-1]=_record(3,"a","b",float("nan"),1)
    with pytest.raises(ValueError,match="nonfinite"):
        summarize_probe(nan)
