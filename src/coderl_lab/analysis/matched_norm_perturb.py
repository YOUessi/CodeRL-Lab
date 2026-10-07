from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_matched_norm_perturbation(
    *,
    base_adapter_dir: Path,
    reference_adapter_dir: Path,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("matched-norm perturbation requires torch+safetensors") from exc

    base_path = base_adapter_dir / "adapter_model.safetensors"
    ref_path = reference_adapter_dir / "adapter_model.safetensors"
    if not base_path.exists() or not ref_path.exists():
        raise FileNotFoundError("base/reference adapter_model.safetensors missing")

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = base_adapter_dir / "adapter_config.json"
    if config_path.exists():
        shutil.copy2(config_path, output_dir / "adapter_config.json")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    output: dict[str, torch.Tensor] = {}
    tensor_rows: list[dict[str, Any]] = []
    global_target2 = 0.0
    global_actual2 = 0.0
    global_dot = 0.0

    with safe_open(base_path, framework="pt", device="cpu") as base_file, safe_open(
        ref_path, framework="pt", device="cpu"
    ) as ref_file:
        base_keys = list(base_file.keys())
        if list(ref_file.keys()) != base_keys:
            raise ValueError("base/reference adapter tensor keys differ")
        metadata = base_file.metadata()

        for key in base_keys:
            base = base_file.get_tensor(key)
            ref = ref_file.get_tensor(key)
            if tuple(base.shape) != tuple(ref.shape):
                raise ValueError(f"shape mismatch for {key}")

            base_f = base.float()
            ref_delta = ref.float() - base_f
            target_norm = float(torch.linalg.vector_norm(ref_delta).item())

            if target_norm == 0.0:
                perturb = torch.zeros_like(base_f)
            else:
                noise = torch.randn(
                    base_f.shape,
                    generator=generator,
                    dtype=torch.float32,
                )
                noise_norm = torch.linalg.vector_norm(noise)
                if float(noise_norm.item()) == 0.0:
                    raise RuntimeError(f"zero random norm for {key}")
                perturb = noise * (target_norm / noise_norm)

            perturbed = (base_f + perturb).to(base.dtype)
            actual_delta = perturbed.float() - base_f
            actual_norm = float(torch.linalg.vector_norm(actual_delta).item())

            output[key] = perturbed.contiguous()

            target2 = target_norm * target_norm
            actual2 = actual_norm * actual_norm
            dot = float((actual_delta.flatten() * ref_delta.flatten()).sum().item())
            global_target2 += target2
            global_actual2 += actual2
            global_dot += dot

            tensor_rows.append(
                {
                    "tensor": key,
                    "target_delta_l2": target_norm,
                    "actual_delta_l2": actual_norm,
                    "absolute_norm_error": abs(actual_norm - target_norm),
                }
            )

    out_path = output_dir / "adapter_model.safetensors"
    save_file(output, str(out_path), metadata=metadata)

    target_global = global_target2 ** 0.5
    actual_global = global_actual2 ** 0.5
    cosine = (
        global_dot / (target_global * actual_global)
        if target_global > 0 and actual_global > 0
        else None
    )

    result = {
        "seed": seed,
        "base_adapter": str(base_adapter_dir),
        "reference_adapter": str(reference_adapter_dir),
        "output_dir": str(output_dir),
        "base_sha256": sha256(base_path),
        "reference_sha256": sha256(ref_path),
        "output_sha256": sha256(out_path),
        "tensor_count": len(tensor_rows),
        "target_global_delta_l2": target_global,
        "actual_global_delta_l2": actual_global,
        "global_norm_ratio": (
            actual_global / target_global if target_global > 0 else 1.0
        ),
        "cosine_with_reference_delta": cosine,
        "max_tensor_norm_error": max(
            (row["absolute_norm_error"] for row in tensor_rows),
            default=0.0,
        ),
        "per_tensor": tensor_rows,
    }
    (output_dir / "matched_norm_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-adapter", type=Path, required=True)
    parser.add_argument("--reference-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = create_matched_norm_perturbation(
        base_adapter_dir=args.base_adapter,
        reference_adapter_dir=args.reference_adapter,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
