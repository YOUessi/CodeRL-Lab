"""Freeze Track A DPO input and bind it to a genuinely trained SFT adapter.

Unlike EXP-004B, this stage trains on UltraFeedback preferences with an
adapter trained on UltraChat. The binding is emitted only after a real SFT
checkpoint exists, and never silently reuses the old MBPP adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl_prompts(path: Path) -> set[str]:
    prompts: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = str(row.get("prompt", ""))
        chosen = str(row.get("chosen", ""))
        rejected = str(row.get("rejected", ""))
        if not prompt.strip() or not chosen.strip() or not rejected.strip() or chosen == rejected:
            raise ValueError(f"malformed preference row at {path}:{line_number}")
        if prompt in prompts:
            raise ValueError(f"duplicate preference prompt at {path}:{line_number}")
        prompts.add(prompt)
    return prompts


def freeze_dpo_config(
    *,
    template_path: Path,
    adapter_dir: Path,
    sft_summary_path: Path,
    manifest_path: Path,
    train_preferences_path: Path,
    validation_preferences_path: Path,
    output_path: Path,
    expected_sft_train_examples: int = 4096,
) -> dict[str, Any]:
    """Fail closed on model, adapter, dataset version, size or split leakage."""
    if output_path.exists():
        raise FileExistsError(f"frozen config already exists: {output_path}")

    cfg = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    summary = json.loads(sft_summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if not isinstance(cfg, dict):
        raise ValueError("invalid DPO template")
    if cfg.get("experiment", {}).get("id") != "TRAIN-007C":
        raise ValueError("only TRAIN-007C DPO template is accepted")
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    init_cfg = cfg["initial_policy"]
    adapter_path = adapter_dir / "adapter_model.safetensors"
    if not adapter_path.is_file():
        raise FileNotFoundError("new SFT adapter weights missing")
    adapter_hash = sha256_file(adapter_path)

    if summary.get("model") != model_cfg.get("name_or_path"):
        raise ValueError("SFT base model differs from DPO model")
    if summary.get("requested_model_revision") != model_cfg.get("revision"):
        raise ValueError("SFT model revision differs from DPO model")
    if summary.get("num_examples") != expected_sft_train_examples:
        raise ValueError("must initialize formal DPO from 4096-example Train-007 SFT")
    if summary.get("saved_adapter_sha256") != adapter_hash:
        raise ValueError("SFT checkpoint SHA does not match SFT training record")
    if summary.get("quantization", {}).get("mode") not in ("nf4", "none"):
        raise ValueError("unrecognized SFT precision pathway")
    if str(init_cfg.get("sft_adapter")) != str(adapter_dir):
        raise ValueError("runtime adapter path mismatches preregistered source")
    if init_cfg.get("expected_sha256") != "REQUIRES_VERIFIED_TRAIN007A_CHECKPOINT":
        raise ValueError("DPO template had an unexpectedly prefilled adapter hash")

    if manifest.get("stage") != "dpo" or manifest.get("track") != "A-posttraining":
        raise ValueError("wrong preference data manifest")
    if manifest.get("full_scan") is not True or manifest.get("prompt_overlap") != 0:
        raise ValueError("smoke or contaminated preference dataset")
    if data_cfg["revision"] != manifest.get("dataset_revision"):
        raise ValueError("preference dataset revision mismatch")
    train = manifest["train"]
    heldout = manifest["validation"]
    if train["rows"] != data_cfg["expected_train_pairs"] or heldout["rows"] != data_cfg["expected_validation_pairs"]:
        raise ValueError("unexpected preference train/eval pair counts")
    if sha256_file(train_preferences_path) != train["sha256"]:
        raise ValueError("DPO train SHA mismatch")
    if sha256_file(validation_preferences_path) != heldout["sha256"]:
        raise ValueError("DPO validation SHA mismatch")

    train_prompts = _read_jsonl_prompts(train_preferences_path)
    validation_prompts = _read_jsonl_prompts(validation_preferences_path)
    if len(train_prompts) != train["rows"] or len(validation_prompts) != heldout["rows"]:
        raise ValueError("preference prompt count mismatch")
    if train_prompts & validation_prompts:
        raise ValueError("DPO train/validation prompt leakage")

    cfg["initial_policy"]["expected_sha256"] = adapter_hash
    cfg["initial_policy"]["verified_sft_summary"] = str(sft_summary_path)
    cfg["data"]["train_sha256"] = train["sha256"]
    cfg["data"]["heldout_sha256"] = heldout["sha256"]
    cfg["data"]["manifest_sha256"] = sha256_file(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return {
        "experiment": "TRAIN-007C",
        "runtime_config": str(output_path),
        "runtime_config_sha256": sha256_file(output_path),
        "sft_adapter_sha256": adapter_hash,
        "training_pairs": train["rows"],
        "heldout_pairs": heldout["rows"],
        "preference_manifest_sha256": cfg["data"]["manifest_sha256"],
        "validated_disjoint": True,
        "new_dpo_training_completed": False,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Freeze DPO config with real SFT weights and verified data")
    p.add_argument("--template", type=Path, default=Path("configs/posttrain_a_dpo_qwen3_1.7b.yaml"))
    p.add_argument("--adapter", type=Path, default=Path("artifacts/posttrain-a/train007a-qlora-ultrachat"))
    p.add_argument("--sft-summary", type=Path, default=Path("artifacts/posttrain-a/train007a-qlora-ultrachat/run_summary.json"))
    p.add_argument("--manifest", type=Path, default=Path("data/generated/posttrain-h4-v1/dpo_manifest.json"))
    p.add_argument("--train-preferences", type=Path, default=Path("data/generated/posttrain-h4-v1/dpo_train.jsonl"))
    p.add_argument("--validation-preferences", type=Path, default=Path("data/generated/posttrain-h4-v1/dpo_validation.jsonl"))
    p.add_argument("--output", type=Path, default=Path("artifacts/posttrain-a/train007c-dpo-ultrafeedback/frozen_config.yaml"))
    a = p.parse_args()
    summary = freeze_dpo_config(
        template_path=a.template,
        adapter_dir=a.adapter,
        sft_summary_path=a.sft_summary,
        manifest_path=a.manifest,
        train_preferences_path=a.train_preferences,
        validation_preferences_path=a.validation_preferences,
        output_path=a.output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
