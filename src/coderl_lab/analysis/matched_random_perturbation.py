from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

from safetensors import safe_open


def build_matched_random_perturbation(
    *,
    base_adapter: Path,
    reference_adapter: Path,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError(
            "matched perturbation requires torch and safetensors"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    output_adapter = output_dir / "adapter_model.safetensors"

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    tensors: dict[str, Any] = {}
    per_tensor: dict[str, Any] = {}
    target_norm2 = 0.0
    actual_norm2 = 0.0
    ref_dot_actual = 0.0

    with safe_open(base_adapter, framework="pt", device="cpu") as base_file:
        with safe_open(
            reference_adapter,
            framework="pt",
            device="cpu",
        ) as ref_file:
            base_keys = list(base_file.keys())
            if list(ref_file.keys()) != base_keys:
                raise ValueError("adapter tensor keys do not match")

            for key in base_keys:
                base = base_file.get_tensor(key)
                ref = ref_file.get_tensor(key)
                if tuple(base.shape) != tuple(ref.shape):
                    raise ValueError(f"shape mismatch for tensor {key}")

                base32 = base.float()
                ref_delta = ref.float() - base32
                target_norm = float(torch.linalg.vector_norm(ref_delta).item())

                if target_norm == 0.0:
                    random_delta = torch.zeros_like(base32)
                else:
                    noise = torch.randn(
                        base32.shape,
                        generator=generator,
                        dtype=torch.float32,
                    )
                    noise_norm = torch.linalg.vector_norm(noise)
                    if float(noise_norm.item()) == 0.0:
                        raise RuntimeError(f"zero random norm for tensor {key}")
                    random_delta = noise * (target_norm / noise_norm)

                perturbed32 = base32 + random_delta
                perturbed = perturbed32.to(dtype=base.dtype).contiguous()
                tensors[key] = perturbed

                actual_delta = perturbed.float() - base32
                actual_norm = float(
                    torch.linalg.vector_norm(actual_delta).item()
                )
                rel_error = (
                    abs(actual_norm - target_norm) / target_norm
                    if target_norm > 0
                    else 0.0
                )

                dot = float(
                    (ref_delta.flatten() * actual_delta.flatten())
                    .sum()
                    .item()
                )
                cosine = None
                if target_norm > 0 and actual_norm > 0:
                    cosine = dot / (target_norm * actual_norm)

                per_tensor[key] = {
                    "target_delta_l2": target_norm,
                    "actual_delta_l2": actual_norm,
                    "relative_norm_error": rel_error,
                    "cosine_to_reference_delta": cosine,
                    "numel": base.numel(),
                }
                target_norm2 += target_norm * target_norm
                actual_norm2 += actual_norm * actual_norm
                ref_dot_actual += dot

    save_file(tensors, str(output_adapter))

    target_l2 = math.sqrt(target_norm2)
    actual_l2 = math.sqrt(actual_norm2)
    global_cosine = None
    if target_l2 > 0 and actual_l2 > 0:
        global_cosine = ref_dot_actual / (target_l2 * actual_l2)

    config_src = base_adapter.parent / "adapter_config.json"
    if config_src.exists():
        shutil.copy2(config_src, output_dir / "adapter_config.json")

    summary = {
        "seed": seed,
        "base_adapter": str(base_adapter),
        "reference_adapter": str(reference_adapter),
        "output_adapter": str(output_adapter),
        "tensor_count": len(per_tensor),
        "target_global_delta_l2": target_l2,
        "actual_global_delta_l2": actual_l2,
        "global_relative_norm_error": (
            abs(actual_l2 - target_l2) / target_l2
            if target_l2 > 0
            else 0.0
        ),
        "global_cosine_to_reference_delta": global_cosine,
        "max_tensor_relative_norm_error": max(
            (row["relative_norm_error"] for row in per_tensor.values()),
            default=0.0,
        ),
        "per_tensor": per_tensor,
    }
    (output_dir / "perturbation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a per-tensor matched-norm random LoRA perturbation"
    )
    parser.add_argument("--base-adapter", type=Path, required=True)
    parser.add_argument("--reference-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_matched_random_perturbation(
        base_adapter=args.base_adapter,
        reference_adapter=args.reference_adapter,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
