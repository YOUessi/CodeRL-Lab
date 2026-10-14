import pytest
import shutil
import subprocess


def _local_evalplus_image_present():
    if shutil.which('docker') is None:
        return False
    try:
        result = subprocess.run(
            ['docker', 'image', 'inspect', 'semaremask-evalplus:latest'],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


requires_pinned_tang_image = pytest.mark.skipif(
    not _local_evalplus_image_present(),
    reason='Requires pinned EvalPlus Docker image on authorized Tang; real Docker integration checked there',
)

from coderl_lab.analysis.exp7c_sandbox_score import (
    bootstrap_deltas, image_sha, sandbox_check,
)


@requires_pinned_tang_image
def test_pinned_sandbox_passes_synthetic_function():
    assert image_sha().startswith("sha256:")
    result=sandbox_check(
        "def plus_one(x):\n    return x + 1",
        "def check(candidate):\n    assert candidate(2) == 3",
        "plus_one",
    )
    assert result == (True, "none")


@requires_pinned_tang_image
def test_pinned_sandbox_rejects_incorrect_answer():
    result=sandbox_check(
        "def plus_one(x):\n    return 5",
        "def check(candidate):\n    assert candidate(2) == 3",
        "plus_one",
    )
    assert result == (False, "AssertionError")


def test_paired_ci_synthetic_identity():
    paired=[{"base":True,"sft":True} for _ in range(164)]
    outcome=bootstrap_deltas(paired,iterations=100,seed=42)
    assert outcome["observed_delta"]==0.0
    assert outcome["ci95_low"]==0.0
    assert outcome["ci95_high"]==0.0


@requires_pinned_tang_image
def test_candidate_main_guard_is_not_executed_when_imported():
    code = (
        "def plus_one(x):\n"
        "    return x+1\n"
        "if __name__ == '__main__':\n"
        "    raise RuntimeError('script execution must not run')\n"
    )
    test = "def check(candidate):\n    assert candidate(2) == 3"
    assert sandbox_check(code, test, "plus_one") == (True, "none")
