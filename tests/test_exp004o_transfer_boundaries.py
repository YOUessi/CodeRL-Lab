from __future__ import annotations
import copy
import json
from pathlib import Path

import pytest

from coderl_lab.analysis.exp004o_transfer_boundaries import (
    build_comparison, wilson_interval,
)


def inputs():
    return {
        letter: json.loads(Path(path).read_text(encoding="utf-8"))
        for letter,path in (
            ("k","results/exp004k/summary.json"),
            ("l","results/exp004l/summary.json"),
            ("m","results/exp004m/transfer_summary.json"),
            ("n","results/exp004n/summary.json"),
        )
    }


def test_frozen_k_l_n_causal_claim_boundary():
    result=build_comparison(**inputs())
    assert result["experiment"]=="EXP-004O"
    assert result["new_model_generation"] is False
    assert result["new_private_test_access"] is False
    assert result["model_tuning_from_lcb_outcomes"] is False

    k=result["cohorts"]["mbpp_k"]
    l=result["cohorts"]["lcb_v6"]
    n=result["cohorts"]["lcb_v5"]
    assert (k["baseline_correct"],k["gate_triggered"],k["wrong_to_correct"],k["correct_to_wrong"])==(207,159,10,2)
    assert (l["baseline_correct"],l["gate_triggered"],l["changed_trajectories"],l["wrong_to_correct"])==(14,148,119,1)
    assert (n["baseline_correct"],n["gate_triggered"],n["changed_trajectories"],n["wrong_to_correct"])==(4,142,117,0)
    assert k["preregistered_primary_success"] is True
    assert l["preregistered_primary_success"] is False
    assert n["preregistered_primary_success"] is False
    assert result["quality_controls"]["lcb_v5_base512_hit_token_cap"] == 133


def test_conditional_changes_not_equivalent_to_accuracy_gains():
    summary=build_comparison(**inputs())
    v5=summary["cohorts"]["lcb_v5"]
    v6=summary["cohorts"]["lcb_v6"]
    assert v5["change_given_gate"] > 0.8
    assert v6["change_given_gate"] > 0.8
    assert v5["rescue_per_changed_trajectory"]["observed_fraction"] == 0
    assert v6["rescue_per_changed_trajectory"]["observed_fraction"] == pytest.approx(1/119)
    assert summary["external_lcb_rescues_per_changed_output_exploratory"]["observed_fraction"] == pytest.approx(1/236)


def test_wilson_bounds_not_zero_just_because_rescues_zero():
    band=wilson_interval(0,117)
    assert band["observed_fraction"]==0
    assert band["wilson95_low"]==0
    assert band["wilson95_high"]>0
    with pytest.raises(ValueError):
        wilson_interval(-1,117)
    with pytest.raises(ValueError):
        wilson_interval(1,0)


def test_fail_closed_on_source_or_count_tampering():
    data=inputs()
    data["n"]["gate_audit"]["gate_changed"]=143
    with pytest.raises(ValueError,match="inconsistent|internally inconsistent"):
        build_comparison(**data)
    data=inputs()
    data["l"]["private_tests_accessed_only_after_freeze"]=False
    with pytest.raises(ValueError,match="unfrozen"):
        build_comparison(**data)


def test_frozen_v5_and_v6_not_conflated():
    s=build_comparison(**inputs())
    assert s["cohorts"]["lcb_v5"]["tasks"] == 167
    assert s["cohorts"]["lcb_v6"]["tasks"] == 175
    assert s["cohorts"]["mbpp_k"]["tasks"] == 500
    assert any("prospective" in t for t in s["limits"])
    assert any("tune" in t for t in s["limits"])
    assert any("posthoc" in t for t in s["limits"])
