from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from coderl_lab.analysis.livecodebench_official_eval import (
    OfficialLiveCodeBenchExecutor,
)


@pytest.mark.parametrize(
    ("docker_returncode", "expected_code"),
    [
        (137, "candidate-resource-limit"),
        (125, "runner-error"),
    ],
)
def test_no_output_resource_oom_vs_infrastructure(
    tmp_path: Path, monkeypatch, docker_returncode: int, expected_code: str
) -> None:
    """Never count a frozen-memory OOM as a platform outage."""
    testing_util = tmp_path / "lcb_runner" / "evaluation" / "testing_util.py"
    testing_util.parent.mkdir(parents=True)
    testing_util.write_text("# fake pinned testing utility\n", encoding="utf-8")

    def fake_run(command, **kwargs):
        if command[:3] == ["docker", "image", "inspect"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        assert command[:2] == ["docker", "run"]
        assert "1g" in command  # Preserve frozen benchmark memory cap.
        return SimpleNamespace(returncode=docker_returncode, stdout="", stderr="")

    monkeypatch.setattr(
        "coderl_lab.analysis.livecodebench_official_eval.shutil.which",
        lambda program: "/usr/bin/docker",
    )
    monkeypatch.setattr(
        "coderl_lab.analysis.livecodebench_official_eval.subprocess.run",
        fake_run,
    )
    executor = OfficialLiveCodeBenchExecutor(livecodebench_repo=tmp_path)
    result = executor.run(
        code="print('test')",
        sample={"input_output": json.dumps({
            "inputs": ["1\n"],
            "outputs": ["1\n"],
            "fn_name": None,
        })},
    )
    assert result.passed is False
    assert result.results == (expected_code,)
    if docker_returncode == 137:
        assert result.metadata["exit_code"] == 137
        assert result.metadata["error"] == "candidate-resource-limit"
