from coderl_lab.analysis.verifier_gated_bottleneck_eval import summarize_rows


def row(
    task,
    *,
    gate,
    eligible,
    base,
    low,
    high,
    always,
    public_pass,
):
    return {
        "task_id": task,
        "gate_triggered": gate,
        "eligible": eligible,
        "baseline_correct": base,
        "gated_low_destabilize_correct": low,
        "gated_high_destabilize_correct": high,
        "always_low_destabilize_correct": always,
        "baseline_public_all_pass": public_pass,
        "gated_low_destabilize_public_all_pass": public_pass,
        "gated_high_destabilize_public_all_pass": public_pass,
        "always_low_destabilize_public_all_pass": public_pass,
        "baseline_raw": "a",
        "gated_low_destabilize_raw": "b" if gate else "a",
        "gated_high_destabilize_raw": "c" if gate else "a",
        "always_low_destabilize_raw": "d" if eligible else "a",
    }


def test_gate_improves_without_touching_public_pass() -> None:
    rows = [
        row(
            "a",
            gate=True,
            eligible=True,
            base=False,
            low=True,
            high=False,
            always=True,
            public_pass=False,
        ),
        row(
            "b",
            gate=False,
            eligible=True,
            base=True,
            low=True,
            high=True,
            always=False,
            public_pass=True,
        ),
    ]
    out = summarize_rows(rows, iterations=200, seed=42)
    assert out["conditions"]["baseline"]["hidden_correct_fraction"] == 0.5
    assert (
        out["conditions"]["gated_low_destabilize"]["hidden_correct_fraction"]
        == 1.0
    )
    assert (
        out["conditions"]["gated_low_destabilize"]["transitions_vs_baseline"][
            "wrong_to_correct"
        ]
        == 1
    )
    assert out["baseline_public_pass_subset"]["gated_low_changed"] == 0
    assert out["baseline_public_pass_subset"]["always_low_changed"] == 1
