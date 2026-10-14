import json

import pytest

from coderl_lab.analysis.exp014_blind_generate import (
    ARMS, blind_packet, output_sha, read_completed, seed,
)


def test_fixed_same_seed_across_arms_and_different_tasks():
    assert len(ARMS)==3
    assert seed("HH/123")==seed("HH/123")
    assert seed("HH/123")!=seed("HH/456")


def test_blind_packet_has_no_arm_identity_and_separate_key(tmp_path):
    tasks=[{"task_id":"HH/0123","messages":[{"role":"user","content":"Tell me a joke."}]}]
    complete={
        ("HH/0123",arm):{"answer":arm.upper()+" anonymous candidate"}
        for arm in ARMS
    }
    manifest=blind_packet(tasks,complete,tmp_path)
    p=[json.loads(x) for x in (tmp_path/"blind_review_packets.jsonl").read_text().splitlines()]
    k=[json.loads(x) for x in (tmp_path/"blind_assignment_key.jsonl").read_text().splitlines()]
    assert manifest["human_review_status"]=="not_reviewed"
    assert len(p)==len(k)==1
    assert set(p[0]["candidate_answers"])=={"A","B","C"}
    assert "mapping" not in p[0]
    assert set(k[0]["mapping"].values())==set(ARMS)
    assert manifest["raw_text_uploaded"] is False


def test_checkpoint_tampering_detection(tmp_path):
    path=tmp_path/"generation_checkpoint.jsonl"
    sample={
        "task_id":"HH/0123","arm":"sft","seed":seed("HH/0123"),
        "answer":"hello","answer_sha256":output_sha("hello")
    }
    path.write_text(json.dumps(sample)+"\n")
    assert len(read_completed(path,{"HH/0123"}))==1
    sample["answer"]="changed"
    path.write_text(json.dumps(sample)+"\n")
    with pytest.raises(ValueError,match="checkpoint"):
        read_completed(path,{"HH/0123"})
