"""EXP-015 exact tokenizer/collator vs historical offset-masked DPO evaluator.

Compares entire 256-frozen UltraFeedback heldout cohort using the actual
TRL 1.14.1 _tokenize and DataCollatorForPreference implementation.
No generated source text or preference answers are ever exported to GitHub.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

MODEL = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
DATA_SHA = "9d8a62dee49e9841add47a2ed27e485926c41e988ac06185ccd23b0df4e2e450"
MAX_LENGTH = 512


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read_frozen(path: Path) -> list[dict]:
    if digest(path) != DATA_SHA:
        raise ValueError("EXP-015 heldout data SHA drift")
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(rows) != 256 or len({r["task_id"] for r in rows}) != 256:
        raise ValueError("exact frozen 256 unique preference pairs required")
    return rows


def old_selected_ids(tokenizer, prompt: str, completion: str) -> tuple[list[int], int]:
    encoding = tokenizer(
        prompt + completion,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
    )
    offsets = encoding["offset_mapping"]
    selected = [
        i for i, (left, right) in enumerate(offsets)
        if i > 0 and right > len(prompt) and right > left
    ]
    return [encoding["input_ids"][i] for i in selected], len(encoding["input_ids"])


def exact_trl_row(tokenizer, prompt: str, chosen: str, rejected: str) -> dict:
    from trl.trainer.dpo_trainer import _tokenize, DataCollatorForPreference

    eos = tokenizer.eos_token
    if not eos:
        raise ValueError("TRL needs non-empty eos token")
    chosen = chosen if chosen.endswith(eos) else chosen + eos
    rejected = rejected if rejected.endswith(eos) else rejected + eos
    prompt_ids = _tokenize(tokenizer, prompt)["input_ids"]
    chosen_full = _tokenize(tokenizer, prompt + chosen)["input_ids"]
    rejected_full = _tokenize(tokenizer, prompt + rejected)["input_ids"]
    pc = chosen_full[:len(prompt_ids)] == prompt_ids
    pr = rejected_full[:len(prompt_ids)] == prompt_ids
    row = {
        "prompt_ids": prompt_ids,
        "chosen_ids": chosen_full[len(prompt_ids):],
        "rejected_ids": rejected_full[len(prompt_ids):],
    }
    collate = DataCollatorForPreference(
        pad_token_id=tokenizer.pad_token_id, max_length=MAX_LENGTH,
        truncation_mode="keep_start",
    )
    batch = collate([row])
    result = {
        "prompt_tokens": len(prompt_ids),
        "prefix_chosen_exact": pc,
        "prefix_rejected_exact": pr,
        "trainer_excludes_long_prompt": len(prompt_ids) >= MAX_LENGTH,
    }
    for name, index in (("chosen", 0), ("rejected", 1)):
        ids = batch["input_ids"][index].tolist()
        completion_mask = batch["completion_mask"][index].tolist()
        attention_mask = batch["attention_mask"][index].tolist()
        selected = [
            ids[i] for i, flag in enumerate(completion_mask)
            if i > 0 and flag and attention_mask[i]
        ]
        untruncated = row[name+"_ids"]
        result[name] = {
            "trainer_scored_token_ids": selected,
            "trainer_scored_tokens": len(selected),
            "trainer_has_eos": bool(selected and selected[-1] == tokenizer.eos_token_id),
            "full_response_tokens_before_truncation": len(untruncated),
        }
    return result


def audit_one(tokenizer, row: dict) -> dict:
    p, c, r = row["prompt"], row["chosen"], row["rejected"]
    trl = exact_trl_row(tokenizer, p, c, r)
    out = {
        "task_id_sha256": hashlib.sha256(str(row["task_id"]).encode()).hexdigest(),
        "prompt_tokens": trl["prompt_tokens"],
        "trainer_excludes_long_prompt": trl["trainer_excludes_long_prompt"],
        "prefix_chosen_exact": trl["prefix_chosen_exact"],
        "prefix_rejected_exact": trl["prefix_rejected_exact"],
    }
    for name, answer in (("chosen", c), ("rejected", r)):
        old, old_input_len = old_selected_ids(tokenizer, p, answer)
        now = trl[name]
        output = {
            "old_scored_tokens": len(old),
            "trl_scored_tokens": now["trainer_scored_tokens"],
            "old_last_token_eos": bool(old and old[-1] == tokenizer.eos_token_id),
            "trl_last_token_eos": now["trainer_has_eos"],
            "same_scored_token_sequence": old == now["trainer_scored_token_ids"],
            "new_eos_added_only_if_survives": (
                now["trainer_has_eos"] and not (old and old[-1] == tokenizer.eos_token_id)
            ),
            "old_input_tokens": old_input_len,
        }
        out[name] = output
    out["both_trl_scored"] = (
        not out["trainer_excludes_long_prompt"]
        and all(out[name]["trl_scored_tokens"] > 0 for name in ("chosen", "rejected"))
    )
    out["both_old_scored"] = all(
        out[name]["old_scored_tokens"] > 0 for name in ("chosen", "rejected")
    )
    out["both_exactly_same_scored_tokens"] = all(
        out[name]["same_scored_token_sequence"] for name in ("chosen", "rejected")
    )
    return out


def summarize(rows: list[dict]) -> dict:
    if len(rows) != 256:
        raise ValueError("all 256 tasks must be accounted for")
    counters = Counter()
    for r in rows:
        for name in ("chosen", "rejected"):
            for name_flag in ("same_scored_token_sequence", "trl_last_token_eos",
                              "old_last_token_eos", "new_eos_added_only_if_survives"):
                if r[name][name_flag]:
                    counters[name+"_"+name_flag] += 1
            counters[name+"_total_old_scored_tokens"] += r[name]["old_scored_tokens"]
            counters[name+"_total_trl_scored_tokens"] += r[name]["trl_scored_tokens"]
        for field in ("trainer_excludes_long_prompt", "both_trl_scored",
                      "both_old_scored", "both_exactly_same_scored_tokens"):
            if r[field]:
                counters[field] += 1
        if not r["prefix_chosen_exact"]:
            counters["prefix_chosen_mismatch"] += 1
        if not r["prefix_rejected_exact"]:
            counters["prefix_rejected_mismatch"] += 1
    return {
        "experiment": "EXP-015-TRL-DPO-PREPROCESSING-AUDIT",
        "status": "completed_cpu_full_256",
        "dataset_sha256": DATA_SHA,
        "model": MODEL, "revision": REVISION,
        "trainer_tokenization": "TRL _tokenize; nonconversational chosen/rejected + eos; DataCollatorForPreference(max_length=512, keep_start)",
        "old_tokenization": "concat(prompt, answer), offset-overlap, no appended EOS, tokenizer truncation max512",
        "rows_total": 256,
        "observed": dict(counters),
        "same_masked_target_tokens_both_answers_fraction": (
            counters["both_exactly_same_scored_tokens"] / 256
        ),
        "note": "This compares exact token IDs/masks, not model quality. EOS and tokenizer prefix-boundary drift must be assessed before interpreting historical raw logP accuracy.",
        "source_private_text_exported": False,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--heldout", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    rows = read_frozen(a.heldout)
    from transformers import AutoTokenizer
    import trl, transformers
    if trl.__version__ != "1.14.1" or transformers.__version__ != "5.18.0":
        raise ValueError("pinned TRL/Transformers version drift")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    outcomes = [audit_one(tokenizer, row) for row in rows]
    summary = summarize(outcomes)
    summary["trl_version"] = trl.__version__
    summary["transformers_version"] = transformers.__version__
    summary["source_script_sha256"] = digest(Path(__file__))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    # Raw token IDs and pref text are never exported; task hashes and counts are safe.
    details = a.output.with_suffix(".details.jsonl")
    for path in (a.output, details):
        if path.exists():
            raise FileExistsError("EXP015 files frozen; refusing overwrite")
    with details.open("x", encoding="utf-8") as f:
        for row in outcomes:
            f.write(json.dumps(row, ensure_ascii=False)+"\n")
    summary["details_sha256"] = digest(details)
    a.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
