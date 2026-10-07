from coderl_lab.data.randomize_preferences import randomize_labels


def test_randomize_half_labels_preserves_texts() -> None:
    rows = [
        {"prompt": f"p{i}", "chosen": f"good{i}", "rejected": f"bad{i}"}
        for i in range(6)
    ]
    out, swapped = randomize_labels(rows, seed=42, swap_fraction=0.5)
    assert len(swapped) == 3
    assert len(out) == 6
    for before, after in zip(rows, out, strict=True):
        assert before["prompt"] == after["prompt"]
        assert sorted([before["chosen"], before["rejected"]]) == sorted(
            [after["chosen"], after["rejected"]]
        )


def test_randomize_is_deterministic() -> None:
    rows = [
        {"prompt": f"p{i}", "chosen": f"a{i}", "rejected": f"b{i}"}
        for i in range(10)
    ]
    a, ia = randomize_labels(rows, seed=7, swap_fraction=0.5)
    b, ib = randomize_labels(rows, seed=7, swap_fraction=0.5)
    assert a == b
    assert ia == ib
