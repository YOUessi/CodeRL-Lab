from coderl_lab.analysis.greedy_first_divergence import first_divergence


def test_first_divergence_same_sequence() -> None:
    assert first_divergence([1, 2, 3], [1, 2, 3]) is None


def test_first_divergence_token_change() -> None:
    assert first_divergence([1, 2, 3], [1, 4, 3]) == 1


def test_first_divergence_length_change() -> None:
    assert first_divergence([1, 2], [1, 2, 3]) == 2
