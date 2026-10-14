"""CPU quality gates for real precommitted single-AI review aggregation."""
import pytest
from coderl_lab.analysis.exp014b_ai_grade_uncertainty import exact_two_sided_sign_test
from coderl_lab.analysis.exp014b_ai_blind_review import aggregate

def test_pairwise_sign_probabilities():
    assert exact_two_sided_sign_test(0,0)==1.0
    assert exact_two_sided_sign_test(22,21)==1.0
    assert exact_two_sided_sign_test(5,0)==pytest.approx(0.0625)
    assert exact_two_sided_sign_test(25,10)==pytest.approx(0.0166738478001)

def test_never_create_false_human_review_from_ai_ratings():
    records=[]
    key=[]
    ratings=[]
    for i in range(64):
        name=f"HH/{i}"
        records.append({"task_id":name,"candidate_answers":{"A":"a","B":"b","C":"c"}})
        key.append({"task_id":name,"mapping":{"A":"sft","B":"bf16_dpo","C":"fp32_dpo"},
                    "answer_sha256":{}})
        scores={"A":3,"B":2,"C":3}
        ratings.append({"index":i,"kind":"AI_SINGLE_REVIEWER_BLIND",
                        "scores":scores,"status":"tied","top_labels":["A","C"]})
    # Empty model answer SHAs must FAIL instead of inventing an aggregate.
    with pytest.raises((ValueError, KeyError)):
        aggregate(packet_rows=records,reviews=ratings,keys=key,
                  keysha="frozen-only-synthetic",freeze={
                     "single_clear_winner":0,"tied":64,
                     "no_acceptable_response":0,"safety_read_restricted":0
                  })
