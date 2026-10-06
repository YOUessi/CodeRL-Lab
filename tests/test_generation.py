from coderl_lab.generation import _sample_batch_ranges, extract_python_code


def test_extracts_fenced_python() -> None:
    text = "解释如下\n```python\ndef solve(x):\n    return x + 1\n```"
    assert extract_python_code(text, "solve") == "def solve(x):\n    return x + 1"


def test_extracts_from_entry_point_and_drops_trailing_prose() -> None:
    text = "答案：\ndef solve(x):\n    return x * 2\n这就是答案。"
    assert extract_python_code(text, "solve") == "def solve(x):\n    return x * 2"


def test_sample_batch_ranges_default_preserves_single_batch() -> None:
    assert _sample_batch_ranges(16, None) == [(0, 16)]


def test_sample_batch_ranges_chunks_large_k() -> None:
    assert _sample_batch_ranges(64, 16) == [
        (0, 16),
        (16, 16),
        (32, 16),
        (48, 16),
    ]


def test_sample_batch_ranges_handles_tail() -> None:
    assert _sample_batch_ranges(34, 16) == [(0, 16), (16, 16), (32, 2)]
