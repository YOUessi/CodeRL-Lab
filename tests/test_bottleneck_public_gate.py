from coderl_lab.analysis.bottleneck_public_gate import summarize_gate


def make_row(task, base, destab, stabilize):
    return {
        "task_id": task,
        "baseline_correct": base,
        "low_destabilize_correct": destab,
        "low_stabilize_correct": stabilize,
    }


def test_public_fail_stratum() -> None:
    rows = [
        make_row("a", False, True, False),
        make_row("a", False, True, False),
        make_row("b", True, True, True),
        make_row("b", True, True, True),
    ]
    ref = {
        "a": {
            "public": {"pass_rate": 0.0},
            "hidden_all_pass": False,
        },
        "b": {
            "public": {"pass_rate": 1.0},
            "hidden_all_pass": True,
        },
    }

    out = summarize_gate(rows, ref, iterations=200, seed=42)
    assert out["gate_quality"]["public_fail_tasks"] == 1
    assert out["gate_quality"]["hidden_wrong_tasks"] == 1
    assert out["gate_quality"]["public_fail_precision_for_hidden_wrong"] == 1.0
    assert out["gate_quality"]["public_fail_recall_of_hidden_wrong"] == 1.0

    fail = out["strata"]["public_fail"]
    assert fail["low_destabilize_delta"] == 1.0
    assert fail["low_destabilize_transitions"]["wrong_to_correct"] == 2
