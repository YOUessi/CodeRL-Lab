from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from coderl_lab.analysis.exp004m_transfer_diagnostics import (
    fisher_exact_two_sided,
    read_frozen_counts,
    render_markdown,
    summarize_transfer,
    wilson_interval,
)

ROOT = Path(__file__).resolve().parents[1]


def load_summaries():
    k = json.loads((ROOT / "results/exp004k/summary.json").read_text())
    l = json.loads((ROOT / "results/exp004l/summary.json").read_text())
    return k, l


def test_ood_readout_preserves_frozen_primary_statistics() -> None:
    k, l = load_summaries()
    out = summarize_transfer(k, l)
    mk = out["metrics"]["mbpp_test_500"]
    ml = out["metrics"]["livecodebench_v6_175"]
    assert (mk["gated"], mk["wrong_gated"], mk["rescued"], mk["harmed"]) == (
        159, 152, 10, 2
    )
    assert (ml["gated"], ml["wrong_gated"], ml["rescued"], ml["harmed"]) == (
        148, 148, 1, 0
    )
    assert mk["rescue_per_wrong_trigger"] == pytest.approx(10 / 152)
    assert ml["rescue_per_wrong_trigger"] == pytest.approx(1 / 148)
    assert out["exploratory_between_domain_comparison"][
        "rescue_per_wrong_trigger_k_minus_l"
    ] == pytest.approx(10 / 152 - 1 / 148)
    assert ml["gate_precision_for_wrong"] == 1.0
    assert mk["net_correct_gain_fraction"] == pytest.approx(8 / 500)
    assert ml["net_correct_gain_fraction"] == pytest.approx(1 / 175)
    assert "不是新模型实验" in render_markdown(out)


def test_fisher_exact_matches_canonical_example() -> None:
    assert fisher_exact_two_sided(1, 9, 11, 3) == pytest.approx(
        0.0027594561852201, rel=1e-7
    )
    assert fisher_exact_two_sided(0, 10, 0, 11) == pytest.approx(1)
    with pytest.raises(ValueError):
        fisher_exact_two_sided(1, -1, 0, 1)


def test_wilson_bounds_and_zero_denominator() -> None:
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0
    with pytest.raises(ValueError):
        wilson_interval(0, 0)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


def test_rejects_hidden_leakage_and_nonfrozen_run() -> None:
    k, l = load_summaries()
    l_bad = copy.deepcopy(l)
    l_bad["frozen_inputs"]["private_tests_accessed"] = True
    with pytest.raises(ValueError, match="leaked hidden tests"):
        read_frozen_counts(k, l_bad)
    l_bad = copy.deepcopy(l)
    l_bad["claim_limit"] = "engineering smoke only"
    with pytest.raises(ValueError, match="175-task"):
        read_frozen_counts(k, l_bad)


def test_rejects_unreconciled_gate_transitions() -> None:
    k, l = load_summaries()
    bad = copy.deepcopy(k)
    bad["gate_triggered_subset"]["wrong_to_correct"] += 1
    with pytest.raises(ValueError, match="disagree"):
        read_frozen_counts(bad, l)
