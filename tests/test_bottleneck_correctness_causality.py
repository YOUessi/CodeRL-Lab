from coderl_lab.analysis.bottleneck_correctness_causality import summarize


def row(
    task,
    arm,
    *,
    ref,
    base,
    low,
    destab,
    high,
    base_exact=False,
    low_exact=False,
    destab_exact=False,
    low_applied=True,
    destab_applied=True,
    high_applied=True,
):
    return {
        "task_id": task,
        "arm": arm,
        "reference_correct": ref,
        "baseline_correct": base,
        "low_stabilize_correct": low,
        "low_destabilize_correct": destab,
        "high_stabilize_correct": high,
        "baseline_exact_reference": base_exact,
        "low_stabilize_exact_reference": low_exact,
        "low_destabilize_exact_reference": destab_exact,
        "high_stabilize_exact_reference": base_exact,
        "low_stabilize_applied": low_applied,
        "low_destabilize_applied": destab_applied,
        "high_stabilize_applied": high_applied,
    }


def test_transition_and_reference_strata() -> None:
    rows = [
        row(
            "a",
            "arm1",
            ref=True,
            base=False,
            low=True,
            destab=False,
            high=False,
            low_exact=True,
        ),
        row(
            "a",
            "arm2",
            ref=True,
            base=True,
            low=True,
            destab=False,
            high=True,
            base_exact=True,
            destab_exact=False,
        ),
        row(
            "b",
            "arm1",
            ref=False,
            base=True,
            low=False,
            destab=True,
            high=True,
            low_exact=True,
        ),
        row(
            "b",
            "arm2",
            ref=False,
            base=False,
            low=False,
            destab=True,
            high=False,
        ),
    ]

    out = summarize(rows, iterations=200, seed=42)

    assert out["eligible_pairs"] == 4
    assert out["eligible_tasks"] == 2
    assert out["conditions"]["low_stabilize"]["transitions"]["wrong_to_correct"] == 1
    assert out["conditions"]["low_stabilize"]["transitions"]["correct_to_wrong"] == 1
    assert out["conditions"]["low_destabilize"]["transitions"]["wrong_to_correct"] == 1
    assert out["conditions"]["low_destabilize"]["transitions"]["correct_to_wrong"] == 1

    assert out["reference_correctness_strata"]["reference_correct"]["pairs"] == 2
    assert out["reference_correctness_strata"]["reference_wrong"]["pairs"] == 2

    rescue = out["trajectory_rescue"]
    assert rescue["pairs"] == 2
    assert rescue["reference_correct"] == 1
    assert rescue["reference_wrong"] == 1
    assert rescue["correctness_rescue"] == 1
    assert rescue["correctness_harm"] == 1


def test_high_margin_control_contrast() -> None:
    rows = [
        row(
            "a",
            "arm1",
            ref=True,
            base=False,
            low=True,
            destab=False,
            high=False,
        ),
        row(
            "b",
            "arm1",
            ref=True,
            base=False,
            low=True,
            destab=False,
            high=False,
        ),
    ]
    out = summarize(rows, iterations=200, seed=42)
    contrast = out["contrasts"]["low_stabilize_minus_high_stabilize"]
    assert contrast["observed_delta"] == 1.0
    assert contrast["ci95_low"] == 1.0
    assert contrast["ci95_high"] == 1.0
