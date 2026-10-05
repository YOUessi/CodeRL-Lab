from coderl_lab.generation import build_prompt, extract_python_code


def test_extracts_fenced_python() -> None:
    text = "解释如下\n```python\ndef solve(x):\n    return x + 1\n```"
    assert extract_python_code(text, "solve") == "def solve(x):\n    return x + 1"


def test_extracts_from_entry_point_and_drops_trailing_prose() -> None:
    text = "答案：\ndef solve(x):\n    return x * 2\n这就是答案。"
    assert extract_python_code(text, "solve") == "def solve(x):\n    return x * 2"


def test_extracts_stdio_program_surrounded_by_prose() -> None:
    text = (
        "解法如下：\n"
        "a, b = map(int, input().split())\n"
        "print(a + b)\n"
        "以上就是答案。"
    )
    assert extract_python_code(text, None) == (
        "a, b = map(int, input().split())\n"
        "print(a + b)"
    )


def test_stdio_prompt_requires_full_program_io() -> None:
    prompt = build_prompt(
        "读取两个整数并输出和。",
        "",
        task_mode="stdin_stdout",
    )
    assert "标准输入" in prompt
    assert "标准输出" in prompt
    assert "完整 Python 程序" in prompt
