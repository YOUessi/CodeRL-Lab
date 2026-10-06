from coderl_lab.datasets.mbpp import (
    build_mbpp_task,
    infer_entry_point,
    starter_code_from_reference,
)


REFERENCE = """def helper(x):
    return x

def square_sum(nums):
    return sum(x * x for x in nums)
"""

TESTS = [
    "assert square_sum([1, 2]) == 5",
    "assert square_sum([]) == 0",
    "assert square_sum([-2]) == 4",
]


def sample_row() -> dict:
    return {
        "task_id": 601,
        "text": "Write a function to sum squared elements.",
        "code": REFERENCE,
        "test_list": TESTS,
        "test_setup_code": "",
        "challenge_test_list": ["assert square_sum([3]) == 9"],
    }


def test_infer_entry_point_ignores_helper() -> None:
    assert infer_entry_point(REFERENCE, TESTS) == "square_sum"


def test_starter_code_does_not_leak_solution() -> None:
    starter = starter_code_from_reference(REFERENCE, "square_sum")
    assert "def square_sum(nums):" in starter
    assert "sum(" not in starter
    assert "pass" in starter


def test_training_split_keeps_hidden_tests() -> None:
    task, sft = build_mbpp_task(
        sample_row(),
        split="train",
        public_test_count=2,
    )
    assert task["entry_point"] == "square_sum"
    assert len(task["public_tests"]) == 2
    assert len(task["hidden_tests"]) == 2
    assert task["hidden_tests"][0]["assertion"] == TESTS[2]
    assert sft is not None
    assert sft["response"] == REFERENCE


def test_test_split_never_emits_sft_target() -> None:
    _, sft = build_mbpp_task(
        sample_row(),
        split="test",
        public_test_count=1,
    )
    assert sft is None
