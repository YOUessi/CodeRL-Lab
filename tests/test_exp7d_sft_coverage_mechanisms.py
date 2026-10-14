import pytest
from coderl_lab.analysis.exp7d_sft_coverage_mechanisms import (
    analyze, candidate_index, exact_mcnemar_two_sided,
    normalized_ast_key, task_rows,
)


def toy_checkpoints():
    generated=[]
    scores=[]
    for i in range(164):
        task_id=f"HumanEval/{i}"
        for arm in ("base","sft"):
            for n in range(4):
                code=f"def f(x):\n    return x+{i+n}"
                correct=(n==0 and (i<3 if arm=="base" else i<5))
                generated.append({
                    "task_id":task_id,"arm":arm,"sample_id":n,
                    "code":code,"code_sha256":f"{task_id}_{arm}_{n}",
                    "output_tokens":9,
                    "entry_point_defined":True,
                })
                scores.append({
                    "task_id":task_id,"arm":arm,"sample_id":n,
                    "code_sha256":f"{task_id}_{arm}_{n}",
                    "passed":correct,
                })
    return generated,scores


def test_task_and_sample_transition_math():
    gens,scores=toy_checkpoints()
    tasks=task_rows(
        candidate_index(gens,generation=True),
        candidate_index(scores,generation=False),
    )
    result=analyze(tasks)
    assert result["arms"]["base"]["solved_at_k"]==[3,3,3,3]
    assert result["arms"]["sft"]["solved_at_k"]==[5,5,5,5]
    assert result["arms"]["base"]["sample_correct"]==3
    assert result["arms"]["sft"]["sample_correct"]==5
    assert result["task_level_solved4_pair_transitions"]=={
        "base_0_sft_0":159,
        "base_0_sft_1":2,
        "base_1_sft_1":3,
    }
    assert result["paired_task_bootstrap"]["solved_at4_fraction"]["mean_paired_delta"]==2/164
    assert result["source_candidate_code_exported"] is False


def test_refuse_duplicate_or_mutated_checkpoint():
    generated,scored=toy_checkpoints()
    with pytest.raises(ValueError,match="duplicate"):
        candidate_index(generated+[generated[0]],generation=True)
    indexed_gen=candidate_index(generated,generation=True)
    indexed_scores=candidate_index(scored,generation=False)
    assert len(task_rows(indexed_gen,indexed_scores))==164
    indexed_scores[("HumanEval/0","base",0)]["code_sha256"]="tampered"
    with pytest.raises(ValueError,match="does not correspond"):
        task_rows(indexed_gen,indexed_scores)


def test_ast_diversity_and_mcnemar_edge_cases():
    assert normalized_ast_key({"code":"def f(x):\n  return x"})==normalized_ast_key(
        {"code":"def f(x):\n    return x"}
    )
    assert normalized_ast_key({"code":"this is not python!"}) is None
    assert exact_mcnemar_two_sided(0,0)==1.0
    assert exact_mcnemar_two_sided(5,8)==exact_mcnemar_two_sided(8,5)
    assert 0<exact_mcnemar_two_sided(5,8)<1.0
