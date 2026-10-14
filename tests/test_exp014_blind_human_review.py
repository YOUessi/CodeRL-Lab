import json

import pytest

from coderl_lab.analysis.exp014_blind_human_review import (
    aggregate, blinded_agreement, read_completed_rater,
)


def write_jsonl(path,values):
    path.write_text("".join(json.dumps(x)+"\n" for x in values))
    return path


def fixture(tmp_path):
    packs=[
        {"task_id":"HH/a","messages":[{"role":"user","content":"hello"}],
         "candidate_answers":{"A":"x","B":"y","C":"z"}},
        {"task_id":"HH/b","messages":[{"role":"user","content":"world"}],
         "candidate_answers":{"A":"1","B":"2","C":"3"}},
    ]
    keys=[
        {"task_id":"HH/a","mapping":{"A":"sft","B":"fp32_dpo","C":"bf16_dpo"}},
        {"task_id":"HH/b","mapping":{"A":"fp32_dpo","B":"bf16_dpo","C":"sft"}},
    ]
    a=[
        {"task_id":"HH/a","rater_id":"person1","winner":"B"},
        {"task_id":"HH/b","rater_id":"person1","winner":"TIE"},
    ]
    b=[
        {"task_id":"HH/a","rater_id":"person2","winner":"B"},
        {"task_id":"HH/b","rater_id":"person2","winner":"A"},
    ]
    return (
        write_jsonl(tmp_path/"pack.jsonl",packs),
        write_jsonl(tmp_path/"key.jsonl",keys),
        write_jsonl(tmp_path/"a.jsonl",a),
        write_jsonl(tmp_path/"b.jsonl",b),
    )


def test_two_actual_raters_and_model_mapping(tmp_path):
    p,k,a,b=fixture(tmp_path)
    x=aggregate(p,k,a,b,expected=2)
    assert x["status"]=="two_human_reviewers_completed"
    assert x["human_agreement"]["tasks"]==2
    assert x["human_agreement"]["raw_agreement"]==0.5
    assert x["human_winner_votes_by_model"]["fp32_dpo"]==3
    assert x["unanimous_wins_by_model"]["fp32_dpo"]==1
    assert x["human_agreement"]["tie_votes"]==1


def test_no_fake_incomplete_reviews_or_same_rater(tmp_path):
    p,k,a,b=fixture(tmp_path)
    empty=write_jsonl(tmp_path/"partial.jsonl",[])
    with pytest.raises(ValueError,match="all frozen"):
        aggregate(p,k,a,empty,expected=2)
    left=[{"task_id":"HH/a","rater_id":"person1","winner":"B"},
          {"task_id":"HH/b","rater_id":"person1","winner":"A"}]
    write_jsonl(b,left)
    with pytest.raises(ValueError,match="distinct"):
        aggregate(p,k,a,b,expected=2)


def test_invalid_winner_rejected(tmp_path):
    p,k,a,b=fixture(tmp_path)
    values=[{"task_id":"HH/a","rater_id":"person1","winner":"E"},
            {"task_id":"HH/b","rater_id":"person1","winner":"B"}]
    write_jsonl(a,values)
    with pytest.raises(ValueError,match="winner"):
        read_completed_rater(a,{"HH/a","HH/b"})
