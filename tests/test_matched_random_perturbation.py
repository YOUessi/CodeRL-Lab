from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
save_file = pytest.importorskip("safetensors.torch").save_file

from coderl_lab.analysis.matched_random_perturbation import (
    build_matched_random_perturbation,
)


def test_matches_reference_tensor_norms(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    ref_dir = tmp_path / "ref"
    out_dir = tmp_path / "out"
    base_dir.mkdir()
    ref_dir.mkdir()

    base_path = base_dir / "adapter_model.safetensors"
    ref_path = ref_dir / "adapter_model.safetensors"

    save_file(
        {
            "a": torch.zeros(8, dtype=torch.float32),
            "b": torch.ones(4, dtype=torch.float32),
        },
        str(base_path),
    )
    save_file(
        {
            "a": torch.ones(8, dtype=torch.float32) * 0.25,
            "b": torch.ones(4, dtype=torch.float32),
        },
        str(ref_path),
    )

    result = build_matched_random_perturbation(
        base_adapter=base_path,
        reference_adapter=ref_path,
        output_dir=out_dir,
        seed=123,
    )

    assert result["tensor_count"] == 2
    assert result["global_relative_norm_error"] < 1e-6
    assert result["max_tensor_relative_norm_error"] < 1e-6
    assert result["per_tensor"]["b"]["actual_delta_l2"] == 0.0


def test_seed_is_reproducible(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    ref_dir = tmp_path / "ref"
    base_dir.mkdir()
    ref_dir.mkdir()
    base_path = base_dir / "adapter_model.safetensors"
    ref_path = ref_dir / "adapter_model.safetensors"

    save_file({"x": torch.zeros(32)}, str(base_path))
    save_file({"x": torch.ones(32)}, str(ref_path))

    a = build_matched_random_perturbation(
        base_adapter=base_path,
        reference_adapter=ref_path,
        output_dir=tmp_path / "a",
        seed=7,
    )
    b = build_matched_random_perturbation(
        base_adapter=base_path,
        reference_adapter=ref_path,
        output_dir=tmp_path / "b",
        seed=7,
    )

    assert a["global_cosine_to_reference_delta"] == pytest.approx(
        b["global_cosine_to_reference_delta"]
    )
