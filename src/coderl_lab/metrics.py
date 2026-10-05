from __future__ import annotations

from math import prod
from typing import Iterable


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased Pass@k estimator used by code-generation benchmarks.

    Args:
        n: number of sampled completions for one task.
        c: number of correct completions among the n samples.
        k: number of samples considered by Pass@k.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= c <= n:
        raise ValueError("c must satisfy 0 <= c <= n")
    if not 1 <= k <= n:
        raise ValueError("k must satisfy 1 <= k <= n")

    if n - c < k:
        return 1.0

    failure_probability = prod(
        (n - c - i) / (n - i)
        for i in range(k)
    )
    return 1.0 - failure_probability


def mean_pass_at_k(task_counts: Iterable[tuple[int, int]], k: int) -> float:
    values = []
    for n, c in task_counts:
        if n < k:
            continue
        values.append(estimate_pass_at_k(n=n, c=c, k=k))
    if not values:
        raise ValueError(f"no task has at least {k} samples")
    return sum(values) / len(values)
