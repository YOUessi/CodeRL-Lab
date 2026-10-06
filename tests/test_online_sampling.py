from itertools import islice

from coderl_lab.sampling.online import (
    OnlineAdaptiveRepeatSampler,
    OnlineBoundaryState,
)


def test_online_state_tracks_transitions() -> None:
    state = OnlineBoundaryState(["a", "b"])
    event1 = state.update_group("a", [0.1, 1.0, 0.1, 1.0])
    assert event1["status_after"] == "mixed"
    assert state.status("a") == "mixed"

    event2 = state.update_group("a", [1.0, 1.0, 1.0, 1.0])
    assert event2["status_before"] == "mixed"
    assert event2["status_after"] == "flat"
    assert state.transition_counts["mixed->flat"] == 1


def test_update_from_batch_requires_contiguous_groups() -> None:
    state = OnlineBoundaryState(["a", "b"])
    stats = state.update_from_batch(
        task_ids=["a", "a", "a", "a", "b", "b", "b", "b"],
        rewards=[0.0, 1.0, 0.0, 1.0, 0.1, 0.1, 0.1, 0.1],
        num_generations=4,
    )
    assert stats["groups"] == 2
    assert stats["mixed_groups"] == 1
    assert stats["flat_groups"] == 1
    assert state.status_counts() == {"unknown": 0, "mixed": 1, "flat": 1}


def test_sampler_uses_newly_discovered_mixed_task() -> None:
    task_ids = ["a", "b", "c", "d"]
    state = OnlineBoundaryState(task_ids)
    sampler = OnlineAdaptiveRepeatSampler(
        task_ids=task_ids,
        state=state,
        mini_repeat_count=2,
        batch_size=2,
        repeat_count=1,
        exploit_fraction=0.5,
        seed=7,
    )

    it = iter(sampler)
    first = list(islice(it, 4))
    first_unique = list(dict.fromkeys(first))
    assert len(first_unique) == 2

    mixed_index = first_unique[0]
    mixed_task = task_ids[mixed_index]
    state.update_group(mixed_task, [0.0, 1.0])

    second = list(islice(it, 4))
    second_unique = set(second)
    assert mixed_index in second_unique

    exploit_events = [
        e for e in state.selection_events if e["mode"] == "exploit"
    ]
    assert exploit_events
    assert exploit_events[-1]["task_id"] == mixed_task


def test_sampler_length_matches_repeat_contract() -> None:
    task_ids = ["a", "b", "c", "d"]
    state = OnlineBoundaryState(task_ids)
    sampler = OnlineAdaptiveRepeatSampler(
        task_ids=task_ids,
        state=state,
        mini_repeat_count=4,
        batch_size=2,
        repeat_count=2,
        exploit_fraction=0.5,
        seed=42,
    )
    assert len(sampler) == 32
