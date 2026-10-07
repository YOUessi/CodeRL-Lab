from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
st = pytest.importorskip("safetensors.torch")

from coderl_lab.analysis.matched_norm_perturb import create_matched_norm_perturbation


def write_adapter(root: Path, tensors: dict[str, torch.Tensor]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    st.save_file(tensors, str(root / "adapter_model.safetensors"))
    (root / "adapter_config.json").write_text("{}\n", encoding="utf-8")


def test_matches_per_tensor_norm_and_is_reproducible(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    ref_dir = tmp_path / "ref"
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"

    base = {
        "a": torch.zeros(8, dtype=torch.float32),
        "b": torch.ones((2, 3), dtype=torch.float32),
    }
    ref = {
        "a": torch.arange(8, dtype=torch.float32) * 0.01,
        "b": base["b"] + 0.02,
    }
    write_adapter(base_dir, base)
    write_adapter(ref_dir, ref)

    s1 = create_matched_norm_perturbation(
        base_adapter_dir=base_dir,
        reference_adapter_dir=ref_dir,
        output_dir=out1,
        seed=101,
    )
    s2 = create_matched_norm_perturbation(
        base_adapter_dir=base_dir,
        reference_adapter_dir=ref_dir,
        output_dir=out2,
        seed=101,
    )

    assert s1["output_sha256"] == s2["output_sha256"]
    assert s1["tensor_count"] == 2
    assert s1["global_norm_ratio"] == pytest.approx(1.0, rel=1e-6, abs=1e-6)

    generated = st.load_file(str(out1 / "adapter_model.safetensors"))
    for key in base:
        expected = torch.linalg.vector_norm(ref[key] - base[key]).item()
        actual = torch.linalg.vector_norm(generated[key] - base[key]).item()
        assert actual == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_different_seeds_change_direction(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    ref_dir = tmp_path / "ref"
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"

    base = {"a": torch.zeros(64, dtype=torch.float32)}
    ref = {"a": torch.ones(64, dtype=torch.float32) * 0.01}
    write_adapter(base_dir, base)
    write_adapter(ref_dir, ref)

    s1 = create_matched_norm_perturbation(
        base_adapter_dir=base_dir,
        reference_adapter_dir=ref_dir,
        output_dir=out1,
        seed=1,
    )
    s2 = create_matched_norm_perturbation(
        base_adapter_dir=base_dir,
        reference_adapter_dir=ref_dir,
        output_dir=out2,
        seed=2,
    )
    assert s1["output_sha256"] != s2["output_sha256"]


def test_scale_changes_norm_not_direction(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    ref_dir = tmp_path / "ref"
    out_half = tmp_path / "half"
    out_full = tmp_path / "full"

    base = {"a": torch.zeros(128, dtype=torch.float32)}
    ref = {"a": torch.linspace(-0.01, 0.01, 128, dtype=torch.float32)}
    write_adapter(base_dir, base)
    write_adapter(ref_dir, ref)

    half = create_matched_norm_perturbation(
        base_adapter_dir=base_dir,
        reference_adapter_dir=ref_dir,
        output_dir=out_half,
        seed=101,
        scale=0.5,
    )
    full = create_matched_norm_perturbation(
        base_adapter_dir=base_dir,
        reference_adapter_dir=ref_dir,
        output_dir=out_full,
        seed=101,
        scale=1.0,
    )

    assert half["scale"] == 0.5
    assert full["scale"] == 1.0
    assert half["actual_global_delta_l2"] == pytest.approx(
        full["actual_global_delta_l2"] * 0.5,
        rel=1e-5,
        abs=1e-7,
    )

    half_tensor = st.load_file(str(out_half / "adapter_model.safetensors"))["a"]
    full_tensor = st.load_file(str(out_full / "adapter_model.safetensors"))["a"]
    half_delta = half_tensor - base["a"]
    full_delta = full_tensor - base["a"]
    cosine = torch.nn.functional.cosine_similarity(
        half_delta.flatten(),
        full_delta.flatten(),
        dim=0,
    ).item()
    assert cosine == pytest.approx(1.0, rel=1e-5, abs=1e-5)


def test_zero_scale_reproduces_base(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    ref_dir = tmp_path / "ref"
    out_dir = tmp_path / "out"

    base = {"a": torch.randn(32, dtype=torch.float32)}
    ref = {"a": base["a"] + 0.01}
    write_adapter(base_dir, base)
    write_adapter(ref_dir, ref)

    result = create_matched_norm_perturbation(
        base_adapter_dir=base_dir,
        reference_adapter_dir=ref_dir,
        output_dir=out_dir,
        seed=101,
        scale=0.0,
    )
    generated = st.load_file(str(out_dir / "adapter_model.safetensors"))["a"]
    assert torch.equal(generated, base["a"])
    assert result["actual_global_delta_l2"] == 0.0
