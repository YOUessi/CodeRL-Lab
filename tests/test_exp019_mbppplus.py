"""EXP-019 MBPP+ frozen source, Docker oracle and same-7B statistical gates."""
import hashlib
import json
import shutil
import subprocess

import pytest

from coderl_lab.datasets.exp019_mbppplus_freeze import (
    select_public_data,SELECTED,
)
from coderl_lab.analysis.exp019_mbpp_docker import (
    IMAGE,IMAGE_ID,image_sha,test_program as sandbox_test_program,
)
from coderl_lab.analysis.exp019_mbpp_score import (
    ARMS,SEEDS,N_DRAWS,ci,produce_metrics,
)


def synthetic_source():
    return [{
        "task_id":i,
        "code":"def f(x): return x + 1",
        "prompt":f"Write a function f that increases the integer by 1. ID {i}.",
        "test_imports":[],
        "test_list":[f"assert f({k}) == {k+1}" for k in range(3)],
        "test":"assert f(-1)==0\nassert f(12)==13",
    } for i in range(378)]


def test_source_selection_has_no_private_augmented_answers():
    public,hidden,reference,stats=select_public_data(synthetic_source())
    assert len(public)==len(hidden)==len(reference)==SELECTED
    assert stats["eligible_before_SHA24"]==378
    assert len(set(p["task_id"] for p in public))==24
    assert set(public[0])=={"task_id","prompt","source_task_id","public_examples_sha256"}
    serialized=json.dumps(public)
    assert "augmented_test" not in serialized
    assert "reference_code" not in serialized
    # Reverse the source order: hash selection is stable regardless of source ordering.
    reverse,_,_,_=select_public_data(list(reversed(synthetic_source())))
    assert [r["task_id"] for r in reverse]==[r["task_id"] for r in public]


def test_task_seed_pairing_and_solved_at2_rates():
    public=[{"task_id":f"MbppPlus/{i}"} for i in range(24)]
    scores={}
    for q in public:
        num=int(q["task_id"].split("/")[-1])
        for seed in SEEDS:
            for sample in range(N_DRAWS):
                for arm in ARMS:
                    ok=(num<8 and sample==0) or (arm=="new_grpo" and num==8 and sample==1)
                    scores[q["task_id"],seed,sample,arm]={
                        "passed":ok,
                        "python_syntax_valid":True,
                        "verdict":"none" if ok else "AssertionError",
                    }
    out=produce_metrics(public,scores)
    assert out["arms"]["sft"]["candidate_pass"]==16
    assert out["arms"]["new_grpo"]["candidate_pass"]==18
    assert out["arms"]["sft"]["solved_at2"]==16
    assert out["arms"]["new_grpo"]["solved_at2"]==18
    assert out["paired_task_bootstrap"]["new_grpo_minus_sft"]["at2"]["observed"]==pytest.approx(1/24)
    assert ci([0]*24,n=100)["observed"]==0.


def installed_image():
    if not shutil.which("docker"):
        return False
    try:
        r=subprocess.run(["docker","image","inspect",IMAGE,"--format","{{.Id}}"],
                         capture_output=True,text=True,timeout=4)
    except (OSError,subprocess.TimeoutExpired):
        return False
    return r.returncode==0 and r.stdout.strip()==IMAGE_ID


@pytest.mark.skipif(not installed_image(),reason="Requires pinned EvalPlus Docker image on authorized Tang")
def test_real_isolated_positive_negative_mbpp_plus():
    assert image_sha()==IMAGE_ID
    test="assert f(-1) == 0\nassert f(5) == 6"
    data={
        "task_id":"MbppPlus/synthetic",
        "augmented_test":test,
        "public_base_test_list":["assert f(0) == 1","assert f(1) == 2","assert f(3) == 4"],
        "test_imports":[],
        "test_sha256":hashlib.sha256(test.encode()).hexdigest(),
    }
    good=sandbox_test_program("def f(x): return x + 1",data)
    bad=sandbox_test_program("def f(x): return x",data)
    assert good=={"passed":True,"error":"none"}
    assert bad["passed"] is False
