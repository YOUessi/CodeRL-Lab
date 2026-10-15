"""TRAIN-017 all arms, 2 seeds, full-score matrix CPU-only quality gates."""
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from coderl_lab.datasets.train017_new_algorithms import MODEL,REVISION
from coderl_lab.analysis.train017_7b_generate import (
    ARMS,SEEDS,TOTAL,samples_seed,checkpoint_records,
)
from coderl_lab.analysis.train017_7b_docker_score import (
    group_scores,paired_ci,
)


def tasks():
    return [{"task_id":f"holdout/test_family/{i}"} for i in range(24)]


def synthetic_matrix():
    generated={}
    scored={}
    for i,task in enumerate(tasks()):
        q=task["task_id"]
        for seed in SEEDS:
            for n in range(4):
                for arm in ARMS:
                    # Construct distinct at1 / solved@4 outcomes in the same 24 tasks
                    expected={
                        "sft":(i<10 and n==0) or (10<=i<15 and n==3),
                        "previous_grpo":(i<9 and n==0) or (9<=i<17 and n==3),
                        "new_grpo":(i<11 and n==0) or (11<=i<18 and n==2),
                    }[arm]
                    key=(q,seed,n,arm)
                    text=f"{q}|{seed}|{n}|{arm}"
                    generated[key]={
                        "task_id":q,"seed":seed,"sample_id":n,"arm":arm,
                        "actual_seed":samples_seed(q,seed,n),
                        "answer":text,
                        "answer_sha256":hashlib.sha256(text.encode()).hexdigest(),
                        "generated_tokens":128,
                    }
                    scored[key]={
                        "task_id":q,"seed":seed,"sample_id":n,"arm":arm,
                        "answer_sha256":generated[key]["answer_sha256"],
                        "all_passed":bool(expected),
                        "reward":1. if expected else .1,
                        "python_valid":True,"entry_valid":True,
                        "verdict":"none" if expected else "wrong_answer",
                    }
    return generated,scored


def test_model_and_new_data_preflight_remains_pinned_to_7b():
    config=yaml.safe_load(Path("configs/train017_7b_grpo_new_algorithms.yaml").read_text())
    assert config["model"]["name_or_path"]==MODEL=="Qwen/Qwen2.5-7B-Instruct"
    assert config["model"]["revision"]==REVISION
    assert config["grpo"]["init_from_sft_adapter"] is True
    assert config["grpo"]["num_generations"]==4
    assert config["grpo"]["per_device_train_batch_size"]==4
    assert config["grpo"]["adapter_trainable_dtype_after_trainer"]=="float32"
    assert config["runs"]["formal_steps"]==32
    assert config["dataset"]["full_train_tasks"]==48
    assert config["dataset"]["holdout_tasks"]==24


def test_576_three_arm_two_seed_matrix_and_sft_new_grpo_at4_not_copy():
    generation,score=synthetic_matrix()
    assert len(generation)==TOTAL==576
    report=group_scores(tasks(),score,generation)
    assert report["status"]=="completed_all576_docker_scored"
    assert report["metrics"]["sft"]["sample0_correct"]==20
    assert report["metrics"]["sft"]["solved4"]==30
    assert report["metrics"]["new_grpo"]["sample0_correct"]==22
    assert report["metrics"]["new_grpo"]["solved4"]==36
    assert report["metrics"]["previous_grpo"]["solved4"]==34
    assert report["pairwise_task_cluster_bootstrap"]["new_grpo_minus_sft"]["at4"]["estimate"]==pytest.approx(3/24)
    assert report["pairwise_task_cluster_bootstrap"]["new_grpo_minus_sft"]["at1"]["estimate"]==pytest.approx(1/24)


def test_candidate_checkpoint_sha_and_sample_identity(tmp_path):
    generation,_=synthetic_matrix()
    examples=list(generation.values())
    path=tmp_path/"checkpoint.jsonl"
    path.write_text(json.dumps(examples[0])+"\n",encoding="utf-8")
    assert len(checkpoint_records(path,{examples[0]["task_id"]}))==1
    bad=dict(examples[0]);bad["answer"]="rewritten"
    path.write_text(json.dumps(bad)+"\n",encoding="utf-8")
    with pytest.raises(ValueError,match="checkpoint"):
        checkpoint_records(path,{examples[0]["task_id"]})


def test_paired_ci_clusters_by_task_not_by_seed_sample():
    x=paired_ci([.5]*24)
    assert x["estimate"]==pytest.approx(.5)
    assert x["ci95_low"]==pytest.approx(.5)
    with pytest.raises(ValueError,match="24 tasks"):
        paired_ci([0.]*48)
