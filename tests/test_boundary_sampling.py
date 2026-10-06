from coderl_lab.sampling.boundary import (
    classify_reward_group,
    strip_hidden_for_training,
)


def test_classify_mixed_group() -> None:
    assert (
        classify_reward_group(
            [0.1, 1.0, 0.1, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        )
        == "mixed"
    )


def test_classify_flat_all_pass() -> None:
    assert (
        classify_reward_group(
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        )
        == "flat_all_pass"
    )


def test_classify_flat_test_fail() -> None:
    assert (
        classify_reward_group(
            [0.1, 0.1, 0.1, 0.1],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        )
        == "flat_test_fail"
    )


def test_boundary_training_row_strips_hidden_tests() -> None:
    task = {
        "task_id": "x",
        "prompt": "task",
        "entry_point": "solve",
        "starter_code": "def solve():\n    pass\n",
        "public_tests": [{"assertion": "assert solve() == 1"}],
        "hidden_tests": [{"assertion": "assert solve() == 2"}],
        "setup_code": "",
        "metadata": {"split": "train"},
    }
    row = strip_hidden_for_training(task)
    assert "public_tests" in row
    assert "hidden_tests" not in row
    assert row["metadata"]["boundary_selected"] is True
