import json

import pytest

from coderl_lab.analysis.exp014_hh_preference import (
    bootstrap, summarize, load_frozen,
)


def scores(index, sft, bf16, fp32):
    # Use chosen logp=margin; rejected logp=0 to simplify synthetic checking.
    r={
        "index":index,"scorable":True,
        "chosen_target_tokens":5,"rejected_target_tokens":5,
        "chosen_truncated":False,"rejected_truncated":False,
    }
    for arm,margin in (("sft",sft),("bf16_dpo",bf16),("fp32_dpo",fp32)):
        r[f"{arm}_chosen_sum_logp"]=margin
        r[f"{arm}_rejected_sum_logp"]=0
        r[f"{arm}_chosen_mean_logp"]=margin/5
        r[f"{arm}_rejected_mean_logp"]=0
    return r


def test_scoring_independent_source_and_pairwise_accuracy_transitions():
    x=[
        scores(0,-1,1,2),
        scores(1,2,-3,-2),
        scores(2,-1,-2,1),
        {"index":3,"scorable":False},
    ]
    out=summarize(x)
    assert out["pairs_total"]==4
    assert out["pairs_scorable"]==3
    assert out["by_model"]["sft"]["raw_sum_logp_chosen_preferred"]==1
    assert out["by_model"]["bf16_dpo"]["raw_sum_logp_chosen_preferred"]==1
    assert out["by_model"]["fp32_dpo"]["raw_sum_logp_chosen_preferred"]==2
    delta=out["comparison"]["fp32_dpo_minus_bf16_dpo"]["sum"]
    assert delta["wrong_to_correct"]==1
    assert delta["correct_to_wrong"]==0
    assert delta["right_minus_left_binary_preference_accuracy"]["observed"]==pytest.approx(1/3)
    assert out["warning"].startswith("Historical public")


def test_bootstrap_fixed_deltas_and_no_empty():
    z=bootstrap([0.5]*7)
    assert z["ci95_low"]==pytest.approx(0.5)
    assert z["ci95_high"]==pytest.approx(0.5)
    with pytest.raises(ValueError):
        bootstrap([])


def test_preference_evaluation_refuses_generation_without_frozen_summary(tmp_path):
    import coderl_lab.analysis.exp014_hh_preference as a
    pub=tmp_path/"pub.jsonl"
    priv=tmp_path/"private.jsonl"
    man=tmp_path/"manifest.json"
    gen=tmp_path/"generation_summary.json"
    for p in (pub,priv,man):p.write_text("{}\n")
    gen.write_text(json.dumps({"status":"generation_started","answers_generated":8}))
    with pytest.raises(ValueError,match="freeze"):
        load_frozen(pub,priv,man,gen,tmp_path/'no_generations.jsonl')
