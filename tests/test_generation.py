from coderl_lab.generation import extract_python_code


def test_extracts_fenced_python() -> None:
    text = "解释如下\n```python\ndef solve(x):\n    return x + 1\n```"
    assert extract_python_code(text, "solve") == "def solve(x):\n    return x + 1"


def test_extracts_from_entry_point_and_drops_trailing_prose() -> None:
    text = "答案：\ndef solve(x):\n    return x * 2\n这就是答案。"
    assert extract_python_code(text, "solve") == "def solve(x):\n    return x * 2"
