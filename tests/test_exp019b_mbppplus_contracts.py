import hashlib
import json
import shutil
import subprocess

import pytest

from coderl_lab.analysis.exp019b_7b_generate import (
    ARMS,DRAWS,sample_seed,read_checkpoint,
)
from coderl_lab.analysis.exp019b_mbppplus_docker_score import (
    IMAGE_ID,extract_code,ast_valid,run_docker,verify_image,summarize,
)


def test_frozen_same_seed_and_public_candidate_checkpoint(tmp_path):
    assert ARMS==("sft","grpo") and DRAWS==4
    assert sample_seed("MBPPPlus/2",0)==sample_seed("MBPPPlus/2",0)
    assert sample_seed("MBPPPlus/2",1)!=sample_seed("MBPPPlus/2",0)
    rows=[{"task_id":"MBPPPlus/2"}]
    src="def solve(x):return x+1"
    rec={"task_id":"MBPPPlus/2","sample_id":0,"arm":"sft",
         "seed":sample_seed("MBPPPlus/2",0),"raw_code":src,
         "code_sha256":hashlib.sha256(src.encode()).hexdigest()}
    file=tmp_path/"checkpoint.jsonl"
    file.write_text(json.dumps(rec)+"\n")
    assert len(read_checkpoint(file,rows))==1
    rec["raw_code"]="tampered"
    file.write_text(json.dumps(rec)+"\n")
    with pytest.raises(ValueError,match="checkpoint"):
        read_checkpoint(file,rows)


def test_format_fence_normalizer_contract():
    ticks=chr(96)*3
    raw=ticks+"python\ndef sum_two(x, y):\n    return x+y\n"+ticks+"\nExplanation"
    src,kind=extract_code(raw)
    assert kind=="first_closed_python_fence"
    assert ast_valid(src,"sum_two")== (True, True)
    assert ast_valid(src,"nonexistent")== (True, False)
    incomplete,kind=extract_code(ticks+"python\nsomething")
    assert kind=="unclosed_leading_fence"
    assert not ast_valid(incomplete,"sum_two")[0]


def test_simple_24_task_paired_aggregation():
    public=[{"task_id":f"MBPPPlus/{i}"} for i in range(24)]
    rows={}
    generated={}
    gold={r["task_id"]:{"passed":True,"reason":"none"} for r in public}
    for r in public:
        q=r["task_id"]
        for arm in ARMS:
            for i in range(DRAWS):
                passed=(i==0) if arm=="sft" else (i<=1)
                rows[q,i,arm]={
                    "passed":passed,"reason":"none" if passed else "AssertionError",
                    "syntax_valid":True,"top_level_entry":True,
                }
                generated[q,i,arm]={"generated_tokens":40}
    result=summarize(public,rows,gold,generated)
    assert result["scores_by_arm"]["sft"]["candidate_correct"]==24
    assert result["scores_by_arm"]["grpo"]["candidate_correct"]==48
    assert result["scores_by_arm"]["sft"]["observed_solved_at4"]==24
    assert result["scores_by_arm"]["grpo"]["observed_solved_at4"]==24
    assert result["paired_task_bootstrap"]["candidate_correct_fraction"]["estimate"]==0.25


def pinned_image_available():
    if shutil.which("docker") is None:return False
    try:
        result=subprocess.run(["docker","image","inspect","semaremask-evalplus:latest",
                               "--format","{{.Id}}"],capture_output=True,
                              text=True,timeout=4)
    except (OSError,subprocess.TimeoutExpired):
        return False
    return result.returncode==0 and result.stdout.strip()==IMAGE_ID


@pytest.mark.skipif(not pinned_image_available(),reason="only Tang has exact pinned EvalPlus Python image")
def test_actual_docker_reference_correctness_and_mismatch():
    assert verify_image()==IMAGE_ID
    code="def times_two(x):\n    return 2*x\n"
    test="assert times_two(5)==10\nassert times_two(-3)==-6"
    assert run_docker(code,test,"times_two")=={"passed":True,"reason":"none"}
    bad="def times_two(x):\n    return x\n"
    assert run_docker(bad,test,"times_two")=={"passed":False,"reason":"AssertionError"}
