"""EXP-019 post-score audit never silently merges source or oracle identities."""
import hashlib
import pytest

from coderl_lab.analysis.exp019_transfer_response_audit import (
    paired,ARMS,SEEDS,N_DRAWS,TOTAL,
)


def fake_frozen():
    gen=[]
    score=[]
    for task_i in range(24):
        for seed in SEEDS:
            for sample in range(N_DRAWS):
                for arm in ARMS:
                    base=f"def solve(): return {task_i}"
                    text=base+("" if arm=="sft" else "\n# preserved same correctness")
                    h=hashlib.sha256(text.encode()).hexdigest()
                    q=f"MbppPlus/{task_i}"
                    row={"task_id":q,"seed":seed,"sample_id":sample,"arm":arm}
                    gen.append({**row,"answer":text,"answer_sha256":h,
                                "generated_tokens":15})
                    correct=(task_i<10)
                    score.append({**row,"answer_sha256":h,
                                  "normalized_code_sha256":h,
                                  "passed":correct,"python_syntax_valid":True,
                                  "verdict":"none" if correct else "AssertionError"})
    return gen,score


def test_full_matrix_can_have_different_text_but_identical_correctness():
    gen,score=fake_frozen()
    res=paired(gen,score)
    assert res["paired_matrix_rows"]==TOTAL
    assert res["by_model"]["sft"]["correct"]==40
    assert res["by_model"]["new_grpo"]["correct"]==40
    comp=res["pairwise"]["new_grpo_minus_sft"]
    assert comp["raw_generation_exact_match"]==0
    assert comp["success_gains"]==0 and comp["success_losses"]==0
    assert comp["status_flips"]=={"old0_new0":56,"old1_new1":40}
    assert res["has_new_GPU_generation"] is False


def test_reject_changed_answer_hash_and_missing_samples():
    gen,score=fake_frozen()
    with pytest.raises(ValueError,match="288"):
        paired(gen[:-1],score)
    gen[0]["answer"]="modified_after_freeze"
    with pytest.raises(ValueError,match="invalid/changed"):
        paired(gen,score)
