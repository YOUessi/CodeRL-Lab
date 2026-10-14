import hashlib
import json
import pytest

from coderl_lab.datasets.train014b_algorithmic_tasks import (
    TRAIN, HELDOUT, expected,inputs, build
)
from coderl_lab.train.train014b_reward_sandbox import (
    score_one,check_code,validate_tests
)
from coderl_lab.analysis.train014b_holdout import sample_seed,ci,verify_checkpoint


def test_training_and_heldout_parameters_truly_disjoint():
    train=build(TRAIN,"train")
    test=build(HELDOUT,"heldout")
    assert len(train)==36 and len(test)==12
    params_train={(x["task_kind"],x["task_id"].rsplit("/",1)[-1]) for x in train}
    params_hold={(x["task_kind"],x["task_id"].rsplit("/",1)[-1]) for x in test}
    assert not params_train & params_hold
    assert all(len(validate_tests(r["test_cases"]))>=19 for r in train)
    assert expected("pairs",0,[0,0,0])==3
    assert expected("subarray_mod",2,[0,0])==3
    assert expected("best_subarray",2,[-3,-1])==-3
    assert expected("distinct_span",2,"abca")==5


def test_partial_correctness_is_grounded_in_real_test_count(monkeypatch):
    import coderl_lab.train.train014b_reward_sandbox as s
    cases=build(TRAIN,"train")[0]["test_cases"]
    n=len(cases)
    monkeypatch.setattr(s,"docker_test",lambda source,tests:{
        "correct":False,"passed":3,"error":"wrong_answer"
    })
    r=score_one("def solve(x):\n    return 0",cases)
    assert r["reward"]==round(.1+.9*3/n,6)
    assert r["tests_passed"]==3 and not r["all_passed"]
    monkeypatch.setattr(s,"docker_test",lambda source,tests:{
        "correct":True,"passed":n,"error":"none"
    })
    assert score_one("def solve(x):\n    return 0",cases)["reward"]==1.0
    assert score_one("not runnable code",cases)["reward"]==0.0


def test_frozen_12_task_4sample_pairing_and_checkpoints(tmp_path):
    task=build(HELDOUT,"heldout")
    sample=task[0]
    assert sample_seed(sample["task_id"],0)==sample_seed(sample["task_id"],0)
    assert sample_seed(sample["task_id"],1)!=sample_seed(sample["task_id"],0)
    assert ci([0]*12)["estimate"]==0.
    f=tmp_path/"generations.jsonl"
    good="def solve(x):\n    return 0"
    entry={"task_id":sample["task_id"],"arm":"base",
           "sample_id":0,"seed":sample_seed(sample["task_id"],0),
           "raw_code":good,"completion_sha256":hashlib.sha256(good.encode()).hexdigest()}
    f.write_text(json.dumps(entry)+"\n")
    assert len(verify_checkpoint(f,task))==1
    entry["raw_code"]="changed"
    f.write_text(json.dumps(entry)+"\n")
    with pytest.raises(ValueError,match="saved completion"):
        verify_checkpoint(f,task)
