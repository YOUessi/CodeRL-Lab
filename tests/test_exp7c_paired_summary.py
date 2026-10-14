import pytest

from coderl_lab.analysis.exp7c_sandbox_score import summarize


def paired_rows():
    rows={}
    for i in range(164):
        qid=f"HumanEval/{i}"
        for sample in range(4):
            for arm in ("base","sft"):
                correct=sample==0 and (i<3 if arm=="base" else i<5)
                rows[(qid,arm,sample)]={
                    "task_id":qid,"arm":arm,"sample_id":sample,
                    "passed":correct,"reason":"none" if correct else "AssertionError",
                }
    return rows


def test_paired_statistics_count_rescues_at_task_level():
    output=summarize(paired_rows())
    assert output["tasks"]==164
    assert output["samples"]==1312
    assert output["conditions"]["base"]["sample_correct"]==3
    assert output["conditions"]["sft"]["sample_correct"]==5
    assert output["conditions"]["base"]["solved_at4"]==3
    assert output["conditions"]["sft"]["solved_at4"]==5
    assert output["paired_transitions"]=={
        "wrong_to_correct":2,
        "correct_to_wrong":0,
        "correct_to_correct":3,
        "wrong_to_wrong":159,
    }
    assert output["sft_minus_base_solved_at4"]["observed_delta"]==2/164


def test_statistics_fail_closed_on_incomplete_task():
    rows=paired_rows()
    del rows[("HumanEval/3","base",2)]
    with pytest.raises(RuntimeError, match="incomplete"):
        summarize(rows)
