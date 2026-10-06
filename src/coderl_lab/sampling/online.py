from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Iterable, Iterator


@dataclass
class TaskSamplingRecord:
    task_id: str
    status: str = "unknown"
    selections: int = 0
    exploit_selections: int = 0
    explore_selections: int = 0
    observations: int = 0
    mixed_observations: int = 0
    flat_observations: int = 0
    last_spread: float | None = None
    last_mean_reward: float | None = None


class OnlineBoundaryState:
    """Mutable online task state shared by sampler and reward function."""

    def __init__(self, task_ids: Iterable[str], *, spread_epsilon: float = 1e-12) -> None:
        ids = [str(x) for x in task_ids]
        if not ids:
            raise ValueError("task_ids cannot be empty")
        if len(set(ids)) != len(ids):
            raise ValueError("task_ids must be unique")
        self.records = {task_id: TaskSamplingRecord(task_id=task_id) for task_id in ids}
        self.spread_epsilon = float(spread_epsilon)
        self.selection_events: list[dict] = []
        self.observation_events: list[dict] = []
        self.transition_counts: dict[str, int] = {}
        self.groups_observed = 0
        self.mixed_groups_observed = 0
        self.flat_groups_observed = 0

    def status(self, task_id: str) -> str:
        return self.records[str(task_id)].status

    def selected_count(self, task_id: str) -> int:
        return self.records[str(task_id)].selections

    def task_ids_with_status(self, status: str) -> list[str]:
        return [task_id for task_id, rec in self.records.items() if rec.status == status]

    def record_selection(self, task_id: str, *, mode: str) -> None:
        task_id = str(task_id)
        if mode not in {"exploit", "explore"}:
            raise ValueError("mode must be exploit or explore")
        rec = self.records[task_id]
        status_before = rec.status
        rec.selections += 1
        if mode == "exploit":
            rec.exploit_selections += 1
        else:
            rec.explore_selections += 1
        self.selection_events.append(
            {
                "order": len(self.selection_events),
                "task_id": task_id,
                "mode": mode,
                "status_before": status_before,
                "selection_number": rec.selections,
            }
        )

    def update_group(self, task_id: str, rewards: list[float]) -> dict:
        task_id = str(task_id)
        if task_id not in self.records:
            raise KeyError(f"unknown task_id: {task_id}")
        if not rewards:
            raise ValueError("rewards cannot be empty")

        rec = self.records[task_id]
        before = rec.status
        spread = float(max(rewards) - min(rewards))
        mean_reward = float(sum(rewards) / len(rewards))
        after = "mixed" if spread > self.spread_epsilon else "flat"

        rec.status = after
        rec.observations += 1
        rec.last_spread = spread
        rec.last_mean_reward = mean_reward
        if after == "mixed":
            rec.mixed_observations += 1
            self.mixed_groups_observed += 1
        else:
            rec.flat_observations += 1
            self.flat_groups_observed += 1
        self.groups_observed += 1

        transition = f"{before}->{after}"
        self.transition_counts[transition] = self.transition_counts.get(transition, 0) + 1
        event = {
            "order": len(self.observation_events),
            "task_id": task_id,
            "status_before": before,
            "status_after": after,
            "spread": spread,
            "mean_reward": mean_reward,
            "rewards": [float(x) for x in rewards],
            "observation_number": rec.observations,
        }
        self.observation_events.append(event)
        return event

    def update_from_batch(
        self,
        *,
        task_ids: list[str],
        rewards: list[float],
        num_generations: int,
    ) -> dict[str, float | int]:
        if num_generations <= 0:
            raise ValueError("num_generations must be positive")
        if len(task_ids) != len(rewards):
            raise ValueError("task_ids and rewards must have equal length")
        if len(task_ids) % num_generations != 0:
            raise ValueError("batch length must be divisible by num_generations")

        mixed = 0
        flat = 0
        groups = 0
        for start in range(0, len(task_ids), num_generations):
            ids = task_ids[start : start + num_generations]
            group_rewards = rewards[start : start + num_generations]
            if len(set(ids)) != 1:
                raise ValueError(
                    "expected contiguous prompt group with one task_id; "
                    f"got {ids}"
                )
            event = self.update_group(ids[0], group_rewards)
            groups += 1
            if event["status_after"] == "mixed":
                mixed += 1
            else:
                flat += 1

        return {
            "groups": groups,
            "mixed_groups": mixed,
            "flat_groups": flat,
            "mixed_fraction": mixed / groups if groups else 0.0,
        }

    def status_counts(self) -> dict[str, int]:
        out = {"unknown": 0, "mixed": 0, "flat": 0}
        for rec in self.records.values():
            out[rec.status] += 1
        return out

    def summary(self) -> dict:
        statuses = self.status_counts()
        total_selections = sum(rec.selections for rec in self.records.values())
        exploit = sum(rec.exploit_selections for rec in self.records.values())
        explore = sum(rec.explore_selections for rec in self.records.values())
        unique_selected = sum(rec.selections > 0 for rec in self.records.values())
        return {
            "num_tasks": len(self.records),
            "status_counts": statuses,
            "groups_observed": self.groups_observed,
            "mixed_groups_observed": self.mixed_groups_observed,
            "flat_groups_observed": self.flat_groups_observed,
            "observed_mixed_fraction": (
                self.mixed_groups_observed / self.groups_observed
                if self.groups_observed
                else 0.0
            ),
            "total_group_selections": total_selections,
            "exploit_selections": exploit,
            "explore_selections": explore,
            "unique_selected_tasks": unique_selected,
            "transition_counts": dict(sorted(self.transition_counts.items())),
        }

    def to_dict(self) -> dict:
        return {
            "summary": self.summary(),
            "records": {
                task_id: asdict(rec)
                for task_id, rec in sorted(self.records.items())
            },
            "selection_events": self.selection_events,
            "observation_events": self.observation_events,
        }


class OnlineAdaptiveRepeatSampler:
    """Lazy adaptive sampler preserving TRL RepeatSampler output structure."""

    def __init__(
        self,
        *,
        task_ids: list[str],
        state: OnlineBoundaryState,
        mini_repeat_count: int,
        batch_size: int,
        repeat_count: int,
        exploit_fraction: float = 0.5,
        seed: int = 42,
    ) -> None:
        if mini_repeat_count <= 0 or batch_size <= 0 or repeat_count <= 0:
            raise ValueError("repeat and batch counts must be positive")
        if not 0.0 <= exploit_fraction <= 1.0:
            raise ValueError("exploit_fraction must be in [0, 1]")
        if len(task_ids) < batch_size:
            raise ValueError("dataset must contain at least batch_size tasks")

        self.task_ids = [str(x) for x in task_ids]
        self.state = state
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.exploit_fraction = exploit_fraction
        self.seed = seed
        self.num_samples = len(task_ids)
        self.num_chunks = self.num_samples // self.batch_size
        self._rng = random.Random(seed)
        self._index_by_task = {task_id: i for i, task_id in enumerate(self.task_ids)}
        if len(self._index_by_task) != len(self.task_ids):
            raise ValueError("task_ids must be unique")

    def _least_selected_choice(
        self,
        candidates: list[str],
        *,
        exclude: set[str],
    ) -> str | None:
        pool = [x for x in candidates if x not in exclude]
        if not pool:
            return None
        min_count = min(self.state.selected_count(x) for x in pool)
        tied = [x for x in pool if self.state.selected_count(x) == min_count]
        return self._rng.choice(tied)

    def _choose_chunk(self) -> list[int]:
        chosen: list[tuple[str, str]] = []
        excluded: set[str] = set()

        exploit_slots = int(round(self.batch_size * self.exploit_fraction))
        exploit_slots = min(self.batch_size, max(0, exploit_slots))

        for _ in range(exploit_slots):
            task_id = self._least_selected_choice(
                self.state.task_ids_with_status("mixed"),
                exclude=excluded,
            )
            if task_id is None:
                break
            chosen.append((task_id, "exploit"))
            excluded.add(task_id)

        while len(chosen) < self.batch_size:
            task_id = self._least_selected_choice(
                self.state.task_ids_with_status("unknown"),
                exclude=excluded,
            )
            if task_id is None:
                task_id = self._least_selected_choice(
                    list(self.state.records),
                    exclude=excluded,
                )
            if task_id is None:
                raise RuntimeError("unable to fill adaptive sampling chunk")
            chosen.append((task_id, "explore"))
            excluded.add(task_id)

        self._rng.shuffle(chosen)
        indexes: list[int] = []
        for task_id, mode in chosen:
            self.state.record_selection(task_id, mode=mode)
            indexes.append(self._index_by_task[task_id])
        return indexes

    def __iter__(self) -> Iterator[int]:
        for _ in range(self.num_chunks):
            chunk = self._choose_chunk()
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return (
            self.num_chunks
            * self.batch_size
            * self.mini_repeat_count
            * self.repeat_count
        )
