import pytest
from coderl_lab.analysis.exp7b_v4_capacity_diagnostics import build_report


def fixture():
    one_point_seven = []
    pub = []
    hidden = []
    for i in range(101):
        qid = f"task_{i:03d}"
        one_point_seven.append({
            "task_id": qid,
            "difficulty": "easy" if i < 22 else "medium",
            "platform": "leetcode" if i < 36 else "atcoder",
            "baseline_correct": i in (0, 1, 2),
            "gated_low_destabilize_correct": i in (0, 1, 2, 3, 4),
        })
        for sample in range(4):
            p = i < 20 and sample == 0
            h = i < 15 and sample == 0
            item = {"question_id": qid, "sample_id": sample,
                    "normalized_sha256": "same-test-hash",
                    "passed": p}
            pub.append(item)
            hidden.append({**item, "passed": h})
    return pub, hidden, one_point_seven


def test_posthoc_diagnostic_uses_pairing_and_task_level():
    pub, hidden, old = fixture()
    summary = build_report(public_rows=pub, private_rows=hidden, exp004p_rows=old)
    counts = summary["counts"]
    assert counts["old_sft512_single_correct"] == 3
    assert counts["old_gated_low_single_correct"] == 5
    assert counts["7b_public_solved4"] == 20
    assert counts["7b_hidden_solved4"] == 15
    assert counts["7b_correct_hidden_completions"] == 15
    assert counts["7b_public_solved4_but_hidden_unsolved"] == 5
    assert counts["7b_new_solved_vs_old_sft_single"] == 12
    assert summary["strata"]["difficulty"]["easy"]["7b_hidden_solved4"] == 15
    assert summary["sample_level_public_vs_private"]["public_pass/hidden_pass"] == 15
    assert summary["sample_level_public_vs_private"]["public_pass/hidden_fail"] == 5
    assert summary["private_test_payload_exported"] is False


def test_refuse_incomplete_or_mismatched_samples():
    pub, hidden, old = fixture()
    with pytest.raises(ValueError, match="404"):
        build_report(public_rows=pub[:-1], private_rows=hidden, exp004p_rows=old)
    with pytest.raises(ValueError, match="different code"):
        hidden[0]["normalized_sha256"] = "changed"
        build_report(public_rows=pub, private_rows=hidden, exp004p_rows=old)
