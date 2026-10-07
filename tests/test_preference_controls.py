from coderl_lab.data.preference_controls import build_noop, build_reverse


ROWS = [
    {"prompt": "p", "chosen": "good", "rejected": "bad"},
    {"prompt": "q", "chosen": "yes", "rejected": "no"},
]


def test_reverse_swaps_all_labels() -> None:
    out = build_reverse(ROWS)
    assert out[0]["chosen"] == "bad"
    assert out[0]["rejected"] == "good"
    assert out[1]["chosen"] == "no"
    assert out[1]["rejected"] == "yes"


def test_noop_makes_identical_pair() -> None:
    out = build_noop(ROWS, side="chosen")
    assert out[0]["chosen"] == out[0]["rejected"] == "good"
    assert out[1]["chosen"] == out[1]["rejected"] == "yes"
