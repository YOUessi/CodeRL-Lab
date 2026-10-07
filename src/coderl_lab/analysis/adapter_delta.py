from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_adapters(a_path: Path, b_path: Path) -> dict[str, Any]:
    try:
        import torch
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("adapter delta analysis requires torch+safetensors") from exc

    a = load_file(str(a_path), device="cpu")
    b = load_file(str(b_path), device="cpu")

    if set(a) != set(b):
        missing_a = sorted(set(b) - set(a))
        missing_b = sorted(set(a) - set(b))
        raise ValueError(
            f"adapter tensor keys differ: missing_a={missing_a[:5]}, "
            f"missing_b={missing_b[:5]}"
        )

    sq_delta = 0.0
    sq_base = 0.0
    max_abs = 0.0
    changed_tensors = 0
    total_tensors = 0
    total_values = 0
    changed_values = 0

    for key in sorted(a):
        ta = a[key]
        tb = b[key]
        if tuple(ta.shape) != tuple(tb.shape):
            raise ValueError(f"shape mismatch for {key}: {ta.shape} vs {tb.shape}")
        da = ta.to(torch.float64)
        db = tb.to(torch.float64)
        diff = db - da
        abs_diff = diff.abs()
        tensor_max = float(abs_diff.max().item()) if diff.numel() else 0.0
        if tensor_max > 0.0:
            changed_tensors += 1
        changed_values += int((abs_diff > 0).sum().item())
        total_values += diff.numel()
        total_tensors += 1
        sq_delta += float((diff * diff).sum().item())
        sq_base += float((da * da).sum().item())
        max_abs = max(max_abs, tensor_max)

    l2 = math.sqrt(sq_delta)
    base_l2 = math.sqrt(sq_base)
    return {
        "a_path": str(a_path),
        "b_path": str(b_path),
        "a_sha256": sha256(a_path),
        "b_sha256": sha256(b_path),
        "tensor_count": total_tensors,
        "changed_tensors": changed_tensors,
        "total_values": total_values,
        "changed_values": changed_values,
        "changed_value_fraction": changed_values / total_values if total_values else 0.0,
        "delta_l2": l2,
        "base_l2": base_l2,
        "relative_l2": l2 / base_l2 if base_l2 else 0.0,
        "max_abs_delta": max_abs,
        "exact_tensor_equality": changed_values == 0,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two LoRA adapter weight files")
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--output", type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = compare_adapters(args.a, args.b)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
