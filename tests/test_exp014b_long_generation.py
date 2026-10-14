import json

from coderl_lab.analysis import exp014b_blind_generate as long
from coderl_lab.analysis import exp014_blind_generate as original


def test_only_long_generation_budget_changes():
    assert original.CONFIG["max_new_tokens"] == 224
    assert long.CONFIG["max_new_tokens"] == 512
    old={k:v for k,v in original.CONFIG.items() if k!="max_new_tokens"}
    new={k:v for k,v in long.CONFIG.items() if k!="max_new_tokens"}
    assert old==new
    assert long.SHA==original.SHA
    assert long.PUBLIC64_SHA==original.PUBLIC64_SHA
    assert long.seed("HH/abc")==original.seed("HH/abc")


def test_new_blinded_packet_valid_without_model_names(tmp_path):
    rows=[{"task_id":"HH/t","messages":[{"role":"user","content":"Help me."}]}]
    outputs={("HH/t",arm):{"answer":arm+" long-answer"} for arm in long.ARMS}
    summary=long.blind_packet(rows,outputs,tmp_path)
    assert summary["human_review_status"]=="not_reviewed"
    packet=json.loads((tmp_path/"blind_review_packets.jsonl").read_text())
    assert set(packet["candidate_answers"])=={"A","B","C"}
    assert "mapping" not in packet
