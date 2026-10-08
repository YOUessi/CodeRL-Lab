from coderl_lab.analysis.heldout_verifier_gated_bottleneck import (
    should_trigger_gate,
)


def test_gate_requires_public_fail_and_low_bottleneck() -> None:
    assert should_trigger_gate(public_pass_rate=0.0, low_position=10)
    assert should_trigger_gate(public_pass_rate=0.5, low_position=10)
    assert not should_trigger_gate(public_pass_rate=1.0, low_position=10)
    assert not should_trigger_gate(public_pass_rate=0.0, low_position=None)
