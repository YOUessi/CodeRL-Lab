"""Frozen tokenization audit for real UltraFeedback reward-model training.

This is a CPU-only preflight: no reward model is trained, no hidden test used,
and the output contains only aggregate counts and hashes (no conversations).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml

from coderl_lab.train.reward_model import (
    pack_reward_tokens,
    validate_preference_snapshot,
)


def audit_split(
    *,
    tokenizer,
    rows: list[dict[str, str]],
    max_length: int,
    max_prompt_tokens: int,
    min_response_tokens: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("empty frozen preference split")
    output: dict[str, Any] = {
        "pairs": len(rows),
        "long_prompt_truncated": 0,
        "chosen_answer_truncated": 0,
        "rejected_answer_truncated": 0,
        "encoded_candidates_identical": 0,
        "unmodified_prompt_retained_in_both": 0,
    }
    prompt_lengths: list[int] = []
    chosen_lengths: list[int] = []
    rejected_lengths: list[int] = []
    effective: list[int] = []

    for row in rows:
        kwargs = {
            "tokenizer": tokenizer,
            "prompt": row["prompt"],
            "max_length": max_length,
            "max_prompt_tokens": max_prompt_tokens,
            "min_response_tokens": min_response_tokens,
        }
        chosen_ids, info_c = pack_reward_tokens(
            **kwargs, completion=row["chosen"],
        )
        rejected_ids, info_r = pack_reward_tokens(
            **kwargs, completion=row["rejected"],
        )
        # Same prompt prefix, even when one answer is much longer.
        psize = info_c["prompt_used_tokens"]
        if psize != info_r["prompt_used_tokens"] or chosen_ids[:psize] != rejected_ids[:psize]:
            raise AssertionError("chosen and rejected did not share identical prompt tokens")
        output["long_prompt_truncated"] += int(info_c["prompt_truncated"])
        output["chosen_answer_truncated"] += int(info_c["answer_truncated"])
        output["rejected_answer_truncated"] += int(info_r["answer_truncated"])
        output["encoded_candidates_identical"] += int(chosen_ids == rejected_ids)
        output["unmodified_prompt_retained_in_both"] += int(not info_c["prompt_truncated"])
        prompt_lengths.append(info_c["prompt_original_tokens"])
        chosen_lengths.append(info_c["answer_original_tokens"])
        rejected_lengths.append(info_r["answer_original_tokens"])
        effective.append(psize)
    for k, vals in (
        ("prompt_original", prompt_lengths),
        ("chosen_original", chosen_lengths),
        ("rejected_original", rejected_lengths),
        ("prompt_used", effective),
    ):
        ordered = sorted(vals)
        output[f"{k}_mean_tokens"] = mean(vals)
        output[f"{k}_median_tokens"] = median(vals)
        output[f"{k}_max_tokens"] = max(vals)
        output[f"{k}_p95_tokens"] = ordered[int(0.95 * (len(vals)-1))]
    return output


def audit_frozen_tokenization(*, config_path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    d = cfg["data"]
    cfg, train, heldout, identity = validate_preference_snapshot(
        config_path=config_path,
        train_path=Path(d["train"]),
        heldout_path=Path(d["heldout"]),
        manifest_path=Path(d["manifest"]),
    )
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["name_or_path"],
        revision=cfg["model"]["revision"],
        trust_remote_code=True,
    )
    t = cfg["training"]
    args = {
        "tokenizer": tokenizer,
        "max_length": int(t["max_length"]),
        "max_prompt_tokens": int(t["max_prompt_tokens"]),
        "min_response_tokens": int(t["min_response_tokens"]),
    }
    result = {
        "experiment": "TRAIN-008A",
        "status": "frozen_data_tokenizer_audit_not_model_training",
        "training_identity": identity,
        "model_revision": cfg["model"]["revision"],
        "representation": {
            k: args[k] for k in ("max_length", "max_prompt_tokens", "min_response_tokens")
        },
        "train": audit_split(rows=train, **args),
        "heldout": audit_split(rows=heldout, **args),
        "private_tests_accessed": False,
        "training_completed": False,
        "caveat": (
            "This only measures token retention and input collisions; "
            "it does not establish preference ranking quality or RLHF gains."
        ),
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/posttrain_a_reward_model_qwen3_1.7b.yaml"))
    p.add_argument("--output", type=Path, default=Path("artifacts/train008a/token_audit.json"))
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("frozen RM token audit output already exists")
    r = audit_frozen_tokenization(config_path=a.config)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "experiment": r["experiment"],
        "status": r["status"],
        "train": r["train"],
        "heldout": r["heldout"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
