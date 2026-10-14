import pytest

from coderl_lab.analysis.exp014_output_audit import summarize
from coderl_lab.analysis.exp014_blind_generate import output_sha


def frozen_outputs():
    rows=[]
    for i in range(64):
        for arm in ("sft","bf16_dpo","fp32_dpo"):
            answer=f"reply for task {i}" if arm!="fp32_dpo" else f"other reply for task {i}"
            rows.append({
                "task_id":f"HH/{i}","arm":arm,
                "answer":answer,"answer_sha256":output_sha(answer),
                "generated_tokens":30,
            })
    return rows


def test_source_label_free_64x3_completeness():
    x=summarize(frozen_outputs())
    assert x["tasks"]==64
    assert x["candidate_answers"]==192
    assert x["by_model"]["sft"]["nonempty_answers"]==64
    assert x["same_exact_answer_task_counts"]["sft_vs_bf16_dpo"]==64
    assert x["same_exact_answer_task_counts"]["sft_vs_fp32_dpo"]==0
    assert x["quality_judgements_produced"]==0


def test_reject_mutated_and_missing_generation():
    rows=frozen_outputs()
    with pytest.raises(ValueError,match="64 tasks"):
        summarize(rows[:-1])
    rows[0]["answer"]="tampered"
    with pytest.raises(ValueError,match="mutated"):
        summarize(rows)
