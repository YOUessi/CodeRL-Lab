import pytest

from coderl_lab.analysis.exp019b_paired_text_audit import summarize
from coderl_lab.analysis.exp019b_7b_generate import ARMS,DRAWS


def make():
    gen={}
    scored={}
    for task in range(24):
        for sample in range(DRAWS):
            for arm in ARMS:
                key=f"MBPPPlus/{task}",sample,arm
                label=sample==0 or (arm=="grpo" and task==0 and sample==1)
                source=f"def f(x):return x+{task+sample}"
                raw_sha=f"SHA_{task}_{sample}_{arm}"
                gen[key]={"code_sha256":raw_sha,"raw_code":source}
                scored[key]={"code_sha256":raw_sha,"passed":label,
                             "reason":"none" if label else "AssertionError"}
    return gen,scored


def test_pairwise_192_source_mutation_and_96_scored_pairing():
    a,b=make()
    r=summarize(a,b)
    assert r["matched_sample_pairs"]==96
    assert r["paired_sample_counts"]["correctness_flip"]==1
    assert r["per_task_aggregate_no_raw_code"]["correctness_flip_tasks"]==1
    assert r["raw_program_source_exported"] is False


def test_reject_incomplete_and_modified_scores():
    a,b=make()
    b.pop(next(iter(b)))
    with pytest.raises(ValueError,match="192"):
        summarize(a,b)
    a,b=make()
    key=next(iter(b))
    b[key]["code_sha256"]="changed"
    with pytest.raises(ValueError,match="checkpoint"):
        summarize(a,b)
