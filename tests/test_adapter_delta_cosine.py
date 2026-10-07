from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from coderl_lab.analysis.adapter_delta_cosine import (
    analyze_delta_directions,
)


def write(path: Path, values: list[float]) -> None:
    save_file(
        {"x": torch.tensor(values, dtype=torch.float32)},
        str(path),
    )


def test_opposite_and_orthogonal_delta_directions(tmp_path: Path) -> None:
    base = tmp_path / "base.safetensors"
    a = tmp_path / "a.safetensors"
    b = tmp_path / "b.safetensors"
    c = tmp_path / "c.safetensors"

    write(base, [0.0, 0.0])
    write(a, [1.0, 0.0])
    write(b, [-1.0, 0.0])
    write(c, [0.0, 2.0])

    out = analyze_delta_directions(
        base_path=base,
        arms={"a": a, "b": b, "c": c},
    )

    assert out["delta_l2"]["a"] == pytest.approx(1.0)
    assert out["delta_l2"]["c"] == pytest.approx(2.0)
    assert out["pairwise_cosine"]["a__b"] == pytest.approx(-1.0)
    assert out["pairwise_cosine"]["a__c"] == pytest.approx(0.0)
