from coderl_lab.train.sft import compute_warmup_steps, prepare_prompt_completion_rows


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


def test_warmup_steps_for_smoke_run() -> None:
    assert compute_warmup_steps(
        num_examples=32,
        epochs=1.0,
        per_device_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_ratio=0.05,
    ) == 1


def test_warmup_steps_for_full_run() -> None:
    assert compute_warmup_steps(
        num_examples=374,
        epochs=3.0,
        per_device_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_ratio=0.05,
    ) == 4
