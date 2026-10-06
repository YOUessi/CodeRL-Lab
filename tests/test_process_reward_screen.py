from coderl_lab.analysis.process_reward_screen import summarize


def row(task, sample, outcome, process, public, dep=True, clean=1.0):
    return {
        "task_id": task,
        "sample_id": sample,
        "outcome_reward": outcome,
        "process_reward": process,
        "public_pass_rate": public,
        "dependency_complete": dep,
        "runtime_clean_rate": clean,
    }


def test_process_reward_can_rescue_outcome_flat_group() -> None:
    rows = [
        row("a", 0, 0.1, 0.25, 0.0, True, 1.0),
        row("a", 1, 0.1, 0.05, 0.0, False, 0.0),
        row("a", 2, 0.1, 0.15, 0.0, True, 0.0),
        row("a", 3, 0.1, 0.25, 0.0, True, 1.0),
        row("b", 0, 1.0, 1.0, 1.0),
        row("b", 1, 1.0, 1.0, 1.0),
        row("b", 2, 1.0, 1.0, 1.0),
        row("b", 3, 1.0, 1.0, 1.0),
    ]
    out = summarize(rows)
    assert out["num_tasks"] == 2
    assert out["outcome_flat_tasks"] == 2
    assert out["process_flat_tasks"] == 1
    assert out["flat_tasks_rescued_by_process_reward"] == 1
    assert out["rescued_zero_public_tasks"] == 1
    assert out["mixed_tasks_collapsed_by_process_reward"] == 0
