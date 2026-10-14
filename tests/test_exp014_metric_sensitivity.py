import pytest

from coderl_lab.analysis.exp014_metric_sensitivity import (
    analyze, exact_mcnemar,
)


def example(index, chosen=6, rejected=3, sft=False, bf=False, fp=False):
    r={
        "index":index,"scorable":True,
        "chosen_target_tokens":chosen,"rejected_target_tokens":rejected,
    }
    for arm,pass_bool in (("sft",sft),("bf16_dpo",bf),("fp32_dpo",fp)):
        for method in ("sum","mean"):
            r[f"{arm}_chosen_{method}_logp"]=1 if pass_bool else -1
            r[f"{arm}_rejected_{method}_logp"]=0
    return r


def test_exact_discordance_p():
    assert exact_mcnemar(0,0)==1.
    assert exact_mcnemar(0,5)==pytest.approx(.0625)
    assert exact_mcnemar(5,0)==pytest.approx(.0625)
    assert exact_mcnemar(1,1)==1.


def test_requires_all_256_scored_rows():
    with pytest.raises(ValueError,match="all256"):
        analyze([example(0)])


def test_length_stratified_accuracy_and_model_transitions():
    rows=[example(i,chosen=10 if i%2==0 else 2,rejected=4,
                  sft=i<100,bf=i<101,fp=i<100)
          for i in range(256)]
    x=analyze(rows)
    assert x["n"]==256
    assert x["length_strata"]["chosen_longer"]["n"]==128
    assert x["length_strata"]["chosen_shorter"]["n"]==128
    diff=x["exact_discordant_pair_tests"]["bf16_dpo_minus_sft"]["sum"]
    assert diff["old_wrong_new_correct"]==1
    assert diff["old_correct_new_wrong"]==0
    assert diff["exact_two_sided_mcnemar_p"]==1.0
    assert x["generated_response_quality_reviewed"] is False
