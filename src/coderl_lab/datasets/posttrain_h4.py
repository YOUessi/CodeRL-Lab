"""Deterministic, streaming SFT/DPO dataset preparation for Track A.

No model/GPU imports at module import time. Original dataset rows, including held-out
records, never enter the Git repository; only hashes and manifests do.

The two HF datasets are intentionally separate training *domains*. This builder does
not pretend that UltraChat SFT plus UltraFeedback DPO is a coding RL benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

ULTRACHAT_REPO = "HuggingFaceH4/ultrachat_200k"
ULTRACHAT_REVISION = "b7fe606ecdbf71e8537946a8d9de5ccf0f6da48b"
ULTRAFEEDBACK_REPO = "HuggingFaceH4/ultrafeedback_binarized"
ULTRAFEEDBACK_REVISION = "daee4cffe8bea93b1cd7874b01c5a6a4ae9dcaf2"
FORMAT_VERSION = "plain-role-v1"
DEFAULT_SEED = 42


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _content(text: Any) -> str:
    if not isinstance(text, str):
        raise ValueError("chat message content must be text")
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("empty message")
    return cleaned


def render_prefix(messages: list[dict[str, Any]]) -> str:
    """Do not use the model's chat template: the Base checkpoint is unchanged.

    For the initial baseline, format all turns before the final assistant turn.
    Mark each turn explicitly and end at the assistant response boundary.
    """
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("prompt prefix must end with a user message")
    parts = []
    for message in messages:
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            raise ValueError("unsupported chat role")
        parts.append(f"### {role.capitalize()}:\n{_content(message.get('content'))}")
    return "\n\n".join(parts) + "\n\n### Assistant:\n"


def _prompt_id(row: dict[str, Any]) -> str:
    value = row.get("prompt_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing prompt_id")
    return value.strip()


def build_sft_record(row: dict[str, Any], split: str) -> tuple[str, dict[str, Any]]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    # First turn is a controlled baseline; preserve multi-turn development as
    # a separate experiment, rather than silently changing the SFT objective.
    if len(messages) < 2 or messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        raise ValueError("SFT row must start with user -> assistant")
    prompt = render_prefix([messages[0]])
    answer = _content(messages[1].get("content"))
    if len(prompt) > 8000 or len(answer) > 12000 or len(answer) < 8:
        raise ValueError("SFT size or quality bound")
    source_id = _prompt_id(row)
    return _sha(prompt), {
        "task_id": f"ultrachat_{split}_{source_id}",
        "prompt": prompt,
        "response": answer + "\n",
        "format": "raw",
        "metadata": {
            "source": ULTRACHAT_REPO,
            "source_revision": ULTRACHAT_REVISION,
            "source_split": split,
            "source_prompt_id": source_id,
            "prompt_sha256": _sha(prompt),
            "format_version": FORMAT_VERSION,
        },
    }


def build_dpo_record(row: dict[str, Any], split: str) -> tuple[str, dict[str, Any]]:
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if not isinstance(chosen, list) or not isinstance(rejected, list):
        raise ValueError("chosen/rejected must be message lists")
    if len(chosen) < 2 or len(chosen) != len(rejected):
        raise ValueError("incompatible preferred and rejected trajectories")
    if chosen[:-1] != rejected[:-1]:
        raise ValueError("chosen/rejected user/context prefixes disagree")
    if chosen[-1].get("role") != "assistant" or rejected[-1].get("role") != "assistant":
        raise ValueError("preference candidates must be assistant messages")
    prompt = render_prefix(chosen[:-1])
    answer = _content(chosen[-1].get("content"))
    other = _content(rejected[-1].get("content"))
    if answer == other or len(prompt) > 8000 or any(len(x) > 12000 or len(x) < 8 for x in (answer, other)):
        raise ValueError("identical or malformed preference candidates")
    source_id = _prompt_id(row)
    return _sha(prompt), {
        "task_id": f"ultrafeedback_{split}_{source_id}",
        "prompt": prompt,
        "chosen": answer + "\n",
        "rejected": other + "\n",
        "metadata": {
            "source": ULTRAFEEDBACK_REPO,
            "source_revision": ULTRAFEEDBACK_REVISION,
            "source_split": split,
            "source_prompt_id": source_id,
            "prompt_sha256": _sha(prompt),
            "format_version": FORMAT_VERSION,
        },
    }


@dataclass(frozen=True)
class Selection:
    records: tuple[dict[str, Any], ...]
    candidates_scanned: int
    valid: int
    rejected: dict[str, int]
    full_scan: bool


def select_subset(
    rows: Iterable[dict[str, Any]],
    *,
    kind: str,
    split: str,
    take: int,
    seed: int = DEFAULT_SEED,
    scan_limit: int | None = None,
) -> Selection:
    """Top-N SHA256 seeded sampling: deterministic, memory O(N), no eval leakage."""
    if kind not in {"sft", "dpo"} or take < 1 or scan_limit is not None and scan_limit < take:
        raise ValueError("invalid subset request")
    builder = build_sft_record if kind == "sft" else build_dpo_record
    rejected: Counter[str] = Counter()
    seen_prompts: set[str] = set()
    heap: list[tuple[int, int, dict[str, Any]]] = []
    count = 0
    valid = 0
    finished = True
    for index, row in enumerate(rows):
        if scan_limit is not None and index >= scan_limit:
            finished = False
            break
        count += 1
        try:
            fingerprint, record = builder(row, split)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            rejected[type(exc).__name__] += 1
            continue
        if fingerprint in seen_prompts:
            rejected["duplicate_prompt"] += 1
            continue
        seen_prompts.add(fingerprint)
        valid += 1
        # Include source and split so the subsampling policy is explicit.
        score = int(_sha(f"{seed}|{kind}|{split}|{fingerprint}"), 16)
        item = (-score, -index, record)
        if len(heap) < take:
            heapq.heappush(heap, item)
        elif score < -heap[0][0]:
            heapq.heapreplace(heap, item)

    if len(heap) != take:
        raise ValueError(f"{kind}/{split}: requested {take}, only {len(heap)} valid records")
    # Order by score, then source index. Crucially, this does not change if
    # the input row order changes (apart from truly duplicated prompt records).
    records = tuple(x[2] for x in sorted(heap, key=lambda x: (-x[0], -x[1])))
    return Selection(records, count, valid, dict(sorted(rejected.items())), finished)


def assert_disjoint(train: Selection, validation: Selection) -> None:
    train_hashes = {row["metadata"]["prompt_sha256"] for row in train.records}
    valid_hashes = {row["metadata"]["prompt_sha256"] for row in validation.records}
    overlap = train_hashes & valid_hashes
    if overlap:
        raise ValueError(f"train/validation prompt leakage: {len(overlap)} shared prompts")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("wb") as out:
        for row in rows:
            blob = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            digest.update(blob)
            out.write(blob)
    return digest.hexdigest()


def _stream_hf(repo: str, revision: str, split: str) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install data extras: pip install -e '.[data]'") from exc
    dataset = load_dataset(repo, split=split, revision=revision, streaming=True)
    yield from dataset


def prepare_dataset(
    *,
    kind: str,
    output_dir: Path,
    train_rows: Iterable[dict[str, Any]],
    validation_rows: Iterable[dict[str, Any]],
    train_size: int,
    validation_size: int,
    seed: int,
    scan_limit: int | None,
) -> dict[str, Any]:
    if kind == "sft":
        repo, revision, train_split, valid_split = (
            ULTRACHAT_REPO, ULTRACHAT_REVISION, "train_sft", "test_sft"
        )
    elif kind == "dpo":
        repo, revision, train_split, valid_split = (
            ULTRAFEEDBACK_REPO, ULTRAFEEDBACK_REVISION, "train_prefs", "test_prefs"
        )
    else:
        raise ValueError("unsupported kind")

    train = select_subset(train_rows, kind=kind, split=train_split,
                          take=train_size, seed=seed, scan_limit=scan_limit)
    validation = select_subset(validation_rows, kind=kind, split=valid_split,
                               take=validation_size, seed=seed, scan_limit=scan_limit)
    assert_disjoint(train, validation)
    train_path = output_dir / f"{kind}_train.jsonl"
    validation_path = output_dir / f"{kind}_validation.jsonl"
    train_sha = write_jsonl(train_path, train.records)
    validation_sha = write_jsonl(validation_path, validation.records)

    manifest = {
        "track": "A-posttraining",
        "format_version": FORMAT_VERSION,
        "dataset_repo": repo,
        "dataset_revision": revision,
        "stage": kind,
        "seed": seed,
        "sampling": "full-stream seeded SHA256 top-N" if scan_limit is None else "first-K seeded SHA256 top-N SMOKE ONLY",
        "full_scan": train.full_scan and validation.full_scan,
        "scan_limit": scan_limit,
        "train": {
            "source_split": train_split, "rows": len(train.records),
            "scanned": train.candidates_scanned, "valid": train.valid,
            "rejected": train.rejected, "sha256": train_sha,
            "file": str(train_path),
        },
        "validation": {
            "source_split": valid_split, "rows": len(validation.records),
            "scanned": validation.candidates_scanned, "valid": validation.valid,
            "rejected": validation.rejected, "sha256": validation_sha,
            "file": str(validation_path),
        },
        "prompt_overlap": 0,
        "test_or_private_examples_in_training": False,
    }
    out = output_dir / f"{kind}_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare frozen Track A SFT/DPO data.")
    parser.add_argument("--stage", choices=("sft", "dpo"), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated/posttrain-h4-v1"))
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--validation-size", type=int)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--smoke-scan-limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "sft":
        repo, revision, train_split, validation_split = (
            ULTRACHAT_REPO, ULTRACHAT_REVISION, "train_sft", "test_sft"
        )
        train_size = args.train_size or 4096
        val_size = args.validation_size or 256
    else:
        repo, revision, train_split, validation_split = (
            ULTRAFEEDBACK_REPO, ULTRAFEEDBACK_REVISION, "train_prefs", "test_prefs"
        )
        train_size = args.train_size or 2048
        val_size = args.validation_size or 256
    result = prepare_dataset(
        kind=args.stage,
        output_dir=args.output_dir,
        train_rows=_stream_hf(repo, revision, train_split),
        validation_rows=_stream_hf(repo, revision, validation_split),
        train_size=train_size,
        validation_size=val_size,
        seed=args.seed,
        scan_limit=args.smoke_scan_limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
