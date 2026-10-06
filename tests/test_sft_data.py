from coderl_lab.train.sft import prepare_prompt_completion_rows


def test_prepare_sft_rows_uses_same_generation_prompt() -> None:
    rows = [
        {
            "task_id": "x",
            "prompt": "Implement add.",
            "starter_code": "def add(a, b):\n    pass\n",
            "response": "def add(a, b):\n    return a + b\n",
        }
    ]
    result = prepare_prompt_completion_rows(rows, seed=42)
    assert len(result) == 1
    assert "Implement add." in result[0]["prompt"]
    assert "def add(a, b):" in result[0]["prompt"]
    assert result[0]["completion"].endswith("\n")
    assert "return a + b" in result[0]["completion"]


def test_max_samples_is_deterministic() -> None:
    rows = [
        {
            "task_id": str(i),
            "prompt": f"task {i}",
            "starter_code": f"def f{i}():\n    pass\n",
            "response": f"def f{i}():\n    return {i}\n",
        }
        for i in range(20)
    ]
    a = prepare_prompt_completion_rows(rows, seed=7, max_samples=5)
    b = prepare_prompt_completion_rows(rows, seed=7, max_samples=5)
    assert a == b
    assert len(a) == 5
