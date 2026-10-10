"""Deterministic train-only prompt selection for online TRAIN-009A PPO smoke.

Never train the policy on UltraFeedback heldout. Never choose prompts using
policy or reward-model success scores. No raw prompts are uploaded to GitHub.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ULTRACHAT_REVISION = "b7fe606ecdbf71e8537946a8d9de5ccf0f6da48b"
ULTRAFEEDBACK_REVISION = "daee4cffe8bea93b1cd7874b01c5a6a4ae9dcaf2"
SFT_TRAIN_SHA = "f868096a21eb37249d06d318fb56ab3b0e3e99c4db442e54c43d6f42b565888b"
SFT_HELDOUT_SHA = "a7f6e07f8569157fe5c3ae0deb65875c3427f9db3d6a5120379e1d9b10dae170"
PREFERENCE_TRAIN_SHA = "945a336244462d43a7c3e70774158a4c40e873b5ee4f6c93d73e0d1e82222b77"
PREFERENCE_HELDOUT_SHA = "9d8a62dee49e9841add47a2ed27e485926c41e988ac06185ccd23b0df4e2e450"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def choose_train_prompts(
    *,
    rows: list[dict[str, Any]],
    excluded_prompt_sha256: set[str],
    n: int,
    seed: int,
) -> list[dict[str, str]]:
    """Only seeded hash-priority order, completely outcome blind."""
    if n <= 0 or not isinstance(seed, int):
        raise ValueError("positive PPO train prompt count and integer seed required")
    eligible: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for row in rows:
        prompt = str(row.get("prompt", ""))
        if not prompt.strip() or not prompt.rstrip().endswith("### Assistant:"):
            raise ValueError("malformed UltraChat training prompt")
        if row.get("format") != "raw":
            raise ValueError("PPO uses the same raw prompt format as SFT")
        fingerprint = sha_prompt(prompt)
        if row.get("metadata", {}).get("prompt_sha256") != fingerprint:
            raise ValueError("SFT source prompt metadata SHA mismatch")
        if fingerprint in seen:
            raise ValueError("duplicate SFT source prompt")
        seen.add(fingerprint)
        if fingerprint in excluded_prompt_sha256:
            continue
        rank = hashlib.sha256(f"{seed}|PPO|{fingerprint}".encode("ascii")).hexdigest()
        eligible.append((rank, fingerprint, prompt))
    eligible.sort(key=lambda e: e[0])
    if len(eligible) < n:
        raise ValueError("insufficient strictly disjoint UltraChat train prompts")
    return [{"prompt_sha256": fp, "prompt": prompt} for _, fp, prompt in eligible[:n]]


def freeze_online_ppo_prompts(*, data_root: Path, n: int, seed: int) -> tuple[list[dict[str, str]], dict[str, Any]]:
    names = {
        "sft_train": "sft_train.jsonl",
        "sft_validation": "sft_validation.jsonl",
        "dpo_train": "dpo_train.jsonl",
        "dpo_validation": "dpo_validation.jsonl",
    }
    pins = {
        "sft_train": SFT_TRAIN_SHA, "sft_validation": SFT_HELDOUT_SHA,
        "dpo_train": PREFERENCE_TRAIN_SHA, "dpo_validation": PREFERENCE_HELDOUT_SHA,
    }
    files = {key: data_root / name for key, name in names.items()}
    if {key: sha_file(path) for key,path in files.items()} != pins:
        raise ValueError("frozen UltraChat/UltraFeedback file SHA differs from source")
    manifest_sft=json.loads((data_root/"sft_manifest.json").read_text(encoding="utf-8"))
    manifest_dpo=json.loads((data_root/"dpo_manifest.json").read_text(encoding="utf-8"))
    if any((
        manifest_sft.get("dataset_revision") != ULTRACHAT_REVISION,
        manifest_dpo.get("dataset_revision") != ULTRAFEEDBACK_REVISION,
        manifest_sft.get("full_scan") is not True,
        manifest_dpo.get("full_scan") is not True,
        manifest_sft.get("stage") != "sft",
        manifest_dpo.get("stage") != "dpo",
        manifest_sft.get("prompt_overlap") != 0,
        manifest_dpo.get("prompt_overlap") != 0,
        manifest_sft["train"]["rows"] != 4096,
        manifest_sft["validation"]["rows"] != 256,
        manifest_dpo["train"]["rows"] != 2048,
        manifest_dpo["validation"]["rows"] != 256,
    )):
        raise ValueError("training snapshots are not the frozen full H4 revisions")
    splits={key:load_rows(path) for key,path in files.items()}
    if any(len(splits[key]) != want for key,want in (
        ("sft_train",4096),("sft_validation",256),
        ("dpo_train",2048),("dpo_validation",256),
    )):
        raise ValueError("actual training or heldout row count drifted")
    validation_sha={sha_prompt(x["prompt"]) for x in splits["dpo_validation"]}
    validation_sha.update(sha_prompt(x["prompt"]) for x in splits["sft_validation"])
    reward_train_sha={sha_prompt(x["prompt"]) for x in splits["dpo_train"]}
    selected=choose_train_prompts(
        rows=splits["sft_train"],
        excluded_prompt_sha256=reward_train_sha|validation_sha,
        n=n,seed=seed,
    )
    chosen_sha=[r["prompt_sha256"] for r in selected]
    if set(chosen_sha)&(reward_train_sha|validation_sha):
        raise AssertionError("RL rollout prompts leaked reward train or heldout labels")
    return selected, {
        "selection_rule": "SHA256(seed|PPO|promptSHA) ascending before outcomes",
        "seed": seed,
        "training_prompt_count": n,
        "selected_prompt_sha256": chosen_sha,
        "prompt_source": "UltraChat SFT train 4096 only",
        "excluded_reward_training_prompt_count": len(reward_train_sha & {
            sha_prompt(x["prompt"]) for x in splits["sft_train"]
        }),
        "selection_overlaps_reward_training": 0,
        "selection_overlaps_any_heldout": 0,
        "file_sha256": pins,
        "source_revisions": {
            "sft": ULTRACHAT_REVISION,
            "reward": ULTRAFEEDBACK_REVISION,
        },
        "no_hidden_or_private_cases_in_training": True,
    }
