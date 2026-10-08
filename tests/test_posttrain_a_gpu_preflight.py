from __future__ import annotations

from types import SimpleNamespace

import pytest

from coderl_lab.train.gpu_preflight import (
    assert_gpu_exclusive,
    classify_gpu_compute_processes,
)


def test_display_process_does_not_block_real_training() -> None:
    result = classify_gpu_compute_processes(
        "2964515, /opt/todesk/bin/ToDesk_Session, 406 MiB\n"
    )
    assert result["safe_for_single_model_job"]
    assert len(result["allowed_display_daemons"]) == 1
    assert result["blocking_processes"] == []


def test_another_model_job_is_blocked() -> None:
    result = classify_gpu_compute_processes(
        "2964515, /opt/todesk/bin/ToDesk_Session, 406 MiB\n"
        "3248181, /tmp/forecastlab-nli-venv/bin/python, 3474 MiB\n"
    )
    assert not result["safe_for_single_model_job"]
    assert result["blocking_processes"][0]["pid"] == "3248181"
    assert result["blocking_processes"][0]["process"] == "python"


def test_unknown_compute_process_fails_closed() -> None:
    result = classify_gpu_compute_processes(
        "3000, /unexpected/custom-driver, 129 MiB\n"
    )
    assert len(result["blocking_processes"]) == 1
    with pytest.raises(ValueError):
        classify_gpu_compute_processes("malformed output")


def test_assert_gpu_exclusive_checks_real_memory(monkeypatch) -> None:
    def fake_run(argv, **kwargs):
        if "--query-compute-apps=pid,process_name,used_gpu_memory" in argv:
            return SimpleNamespace(stdout="1, /opt/todesk/bin/ToDesk_Session, 400 MiB\n")
        return SimpleNamespace(stdout="14500\n")

    monkeypatch.setattr("coderl_lab.train.gpu_preflight.subprocess.run", fake_run)
    result = assert_gpu_exclusive(min_free_mib=11000)
    assert result["free_memory_mib"] == 14500
    with pytest.raises(RuntimeError, match="not enough free"):
        assert_gpu_exclusive(min_free_mib=16000)


def test_assert_gpu_exclusive_never_kills_other_processes(monkeypatch) -> None:
    def fake_run(argv, **kwargs):
        return SimpleNamespace(
            stdout="22, /tmp/project/bin/python, 3000 MiB\n"
        )

    monkeypatch.setattr("coderl_lab.train.gpu_preflight.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="GPU occupied"):
        assert_gpu_exclusive(min_free_mib=3000)
