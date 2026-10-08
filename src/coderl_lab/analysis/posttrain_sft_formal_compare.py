"""Compare actual, checksum-matched TRAIN-007A NF4 vs TRAIN-007B BF16.

This is a *descriptive* matched-budget comparison, never a claim of
statistically significant task-level quality improvement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def last_eval_loss(log: list[dict[str, Any]]) -> float:
    values = [float(x["eval_loss"]) for x in log if "eval_loss" in x]
    if not values or not math.isfinite(values[-1]):
        raise ValueError("missing/nonfinite heldout eval_loss")
    return values[-1]


def actual_optimizer_steps(log: list[dict[str, Any]]) -> int:
    steps = [int(x["step"]) for x in log if "step" in x]
    if not steps or max(steps) <= 0:
        raise ValueError("real GPU optimizer steps not logged")
    return max(steps)


def verify_run(
    *,
    checkpoint_dir: Path, config_path: Path,
    manifest: dict[str, Any], quant_mode: str,
) -> dict[str, Any]:
    if quant_mode not in ("nf4", "none"):
        raise ValueError("invalid quant mode")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    summary = load_json(checkpoint_dir / "run_summary.json")
    history = json.loads((checkpoint_dir / "log_history.json").read_text(encoding="utf-8"))
    weights = checkpoint_dir / "adapter_model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError("trained adapter not saved")
    digest = sha256(weights)
    if any((
        summary.get("model") != config["model"]["name_or_path"],
        summary.get("requested_model_revision") != config["model"]["revision"],
        summary.get("quantization", {}).get("mode") != quant_mode,
        config["quantization"]["mode"] != quant_mode,
        summary.get("num_examples") != manifest["train"]["rows"],
        summary.get("num_heldout_validation_examples") != manifest["validation"]["rows"],
        summary.get("train_data_sha256") != manifest["train"]["sha256"],
        summary.get("heldout_data_sha256") != manifest["validation"]["sha256"],
        summary.get("saved_adapter_sha256") != digest,
        not summary.get("cuda_available"),
        summary.get("seed") != config["training"]["seed"],
    )):
        raise ValueError("SFT run provenance/adapter does not match frozen data")
    for key in ("r", "alpha", "dropout", "target_modules"):
        if summary["lora"][key] != config["lora"][key]:
            raise ValueError(f"run LoRA config mismatch: {key}")
    loss = float(summary["metrics"]["train_loss"])
    eval_loss = last_eval_loss(history)
    time = float(summary["elapsed_seconds_wall"])
    peak = int(summary["gpu"]["peak_reserved_bytes"])
    if not all(math.isfinite(x) for x in (loss, eval_loss, time)) or min(loss, eval_loss, time, peak) <= 0:
        raise ValueError("invalid actual run metrics")
    return {
        "optimizer_steps": actual_optimizer_steps(history),
        "train_loss": loss, "eval_loss": eval_loss,
        "elapsed_seconds_wall": time,
        "gpu_peak_reserved_bytes": peak,
        "adapter_sha256": digest,
        "train_data_sha256": summary["train_data_sha256"],
        "heldout_data_sha256": summary["heldout_data_sha256"],
        "num_train_inputs": summary["num_examples"],
        "num_heldout_inputs": summary["num_heldout_validation_examples"],
        "seed": summary["seed"],
        "dtype": summary["dtype"],
        "quant_mode": quant_mode,
    }


def compare_formal_runs(
    *,
    nf4_config_path: Path, bf16_config_path: Path, manifest_path: Path,
    nf4_dir: Path, bf16_dir: Path,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    cfg_a = yaml.safe_load(nf4_config_path.read_text(encoding="utf-8"))
    cfg_b = yaml.safe_load(bf16_config_path.read_text(encoding="utf-8"))
    if any((
        manifest.get("stage") != "sft",
        manifest.get("track") != "A-posttraining",
        manifest.get("full_scan") is not True,
        manifest.get("prompt_overlap") != 0,
        manifest["train"]["rows"] != 4096,
        manifest["validation"]["rows"] != 256,
    )):
        raise ValueError("requires frozen full 4096/256 dataset manifest")
    for key in ("model", "data", "lora"):
        if cfg_a[key] != cfg_b[key]:
            raise ValueError(f"unmatched {key} config")
    train_a = dict(cfg_a["training"])
    train_b = dict(cfg_b["training"])
    train_a.pop("output_dir", None)
    train_b.pop("output_dir", None)
    if train_a != train_b:
        raise ValueError("unmatched SFT training hyperparameters")
    if cfg_a["quantization"]["mode"] != "nf4" or cfg_b["quantization"]["mode"] != "none":
        raise ValueError("incorrect control arms")
    a = verify_run(
        checkpoint_dir=nf4_dir, config_path=nf4_config_path,
        manifest=manifest, quant_mode="nf4",
    )
    b = verify_run(
        checkpoint_dir=bf16_dir, config_path=bf16_config_path,
        manifest=manifest, quant_mode="none",
    )
    if a["optimizer_steps"] != b["optimizer_steps"]:
        raise ValueError("matched-budget runs have different optimizer step counts")
    if a["train_data_sha256"] != b["train_data_sha256"] or (
        a["heldout_data_sha256"] != b["heldout_data_sha256"]
    ):
        raise ValueError("matched runs used different frozen data")
    return {
        "experiment": "TRAIN-007A-vs-007B",
        "comparison_type": "two real matched-seed, matched-data GPU training runs",
        "formal_data": True,
        "data": {
            "train_rows": 4096, "heldout_rows": 256,
            "train_sha256": manifest["train"]["sha256"],
            "heldout_sha256": manifest["validation"]["sha256"],
        },
        "nf4_qlora": a, "bf16_lora": b,
        "differences_bf16_minus_nf4": {
            "train_loss": b["train_loss"] - a["train_loss"],
            "heldout_eval_loss": b["eval_loss"] - a["eval_loss"],
            "peak_reserved_bytes": b["gpu_peak_reserved_bytes"] - a["gpu_peak_reserved_bytes"],
            "elapsed_seconds_wall": b["elapsed_seconds_wall"] - a["elapsed_seconds_wall"],
        },
        "relative_memory_saving_nf4": 1 - (
            a["gpu_peak_reserved_bytes"] / b["gpu_peak_reserved_bytes"]
        ),
        "limitations": [
            "Single seed, not a statistically powered quality result.",
            "Quantized and unquantized models use different numerical compute paths.",
            "Train input rows can differ from the effective tokenized examples after fully masked samples are dropped.",
            "Loss comparison does not establish task-level generation accuracy.",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path("data/generated/posttrain-h4-v1/sft_manifest.json"))
    p.add_argument("--nf4-config", type=Path, default=Path("configs/posttrain_a_qlora_qwen3_1.7b.yaml"))
    p.add_argument("--bf16-config", type=Path, default=Path("configs/posttrain_a_lora_qwen3_1.7b.yaml"))
    p.add_argument("--nf4-run", type=Path, default=Path("artifacts/posttrain-a/train007a-qlora-ultrachat"))
    p.add_argument("--bf16-run", type=Path, default=Path("artifacts/posttrain-a/train007b-lora-ultrachat"))
    p.add_argument("--output", type=Path, default=Path("artifacts/posttrain-a/train007a-vs-007b/comparison.json"))
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("formal comparison already exists; refusing silent overwrite")
    result = compare_formal_runs(
        nf4_config_path=a.nf4_config, bf16_config_path=a.bf16_config,
        manifest_path=a.manifest, nf4_dir=a.nf4_run, bf16_dir=a.bf16_run,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
