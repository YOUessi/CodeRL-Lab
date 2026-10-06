from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


DATASET_ID = "google-research-datasets/mbpp"
DATASET_CONFIG = "full"
EXPECTED_SPLIT_SIZES = {
    "train": 374,
    "validation": 90,
    "test": 500,
    "prompt": 10,
}


def _function_names(reference_code: str) -> list[str]:
    tree = ast.parse(reference_code)
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _called_function_names(assertion: str) -> list[str]:
    try:
        tree = ast.parse(assertion)
    except SyntaxError:
        return []

    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.append(node.func.id)
    return names


def infer_entry_point(reference_code: str, assertions: Iterable[str]) -> str:
    """Infer the benchmark target function without relying on task wording.

    Prefer a function defined by the canonical solution and actually called by
    an assertion. This avoids choosing wrappers such as set(...) or len(...).
    """
    defined = _function_names(reference_code)
    if not defined:
        raise ValueError("reference code defines no top-level function")

    defined_set = set(defined)
    for assertion in assertions:
        for name in _called_function_names(assertion):
            if name in defined_set:
                return name

    if len(defined) == 1:
        return defined[0]

    raise ValueError(
        "could not infer a unique entry point from canonical code and tests: "
        + ", ".join(defined)
    )


def starter_code_from_reference(reference_code: str, entry_point: str) -> str:
    """Expose only the target signature, never the canonical implementation."""
    tree = ast.parse(reference_code)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == entry_point:
                signature = ast.unparse(node.args)
                prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                return f"{prefix} {entry_point}({signature}):\n    pass\n"
    return f"def {entry_point}(*args, **kwargs):\n    pass\n"


def _assertion_case(assertion: str, index: int, group: str) -> dict[str, Any]:
    return {
        "name": f"{group}_{index:02d}",
        "assertion": assertion,
    }


def build_mbpp_task(
    row: dict[str, Any],
    *,
    split: str,
    public_test_count: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Convert one MBPP row into CodeRL-Lab task + optional SFT record."""
    base_tests = [str(x) for x in row.get("test_list", []) if str(x).strip()]
    challenge_tests = [
        str(x) for x in row.get("challenge_test_list", []) if str(x).strip()
    ]
    if len(base_tests) < 2:
        raise ValueError(
            f"MBPP task {row.get('task_id')} has fewer than two base tests"
        )

    # Always reserve at least one original test as hidden.
    public_count = max(1, min(public_test_count, len(base_tests) - 1))
    public_assertions = base_tests[:public_count]
    hidden_assertions = base_tests[public_count:] + challenge_tests

    reference_code = str(row["code"])
    entry_point = infer_entry_point(reference_code, base_tests + challenge_tests)
    starter_code = starter_code_from_reference(reference_code, entry_point)

    task_id = f"mbpp_{split}_{int(row['task_id']):04d}"
    prompt = (
        str(row["text"]).strip()
        + f"\n\nRequired function name: {entry_point}"
    )

    task = {
        "task_id": task_id,
        "prompt": prompt,
        "entry_point": entry_point,
        "starter_code": starter_code,
        "setup_code": str(row.get("test_setup_code", "") or ""),
        "public_tests": [
            _assertion_case(code, i, "public")
            for i, code in enumerate(public_assertions)
        ],
        "hidden_tests": [
            _assertion_case(code, i, "hidden")
            for i, code in enumerate(hidden_assertions)
        ],
        "metadata": {
            "source": DATASET_ID,
            "source_task_id": int(row["task_id"]),
            "split": split,
            "public_test_count": len(public_assertions),
            "hidden_test_count": len(hidden_assertions),
            "challenge_test_count": len(challenge_tests),
        },
    }

    sft = None
    if split == "train":
        sft = {
            "task_id": task_id,
            "prompt": prompt,
            "starter_code": starter_code,
            "response": reference_code,
            "metadata": {
                "source": DATASET_ID,
                "source_task_id": int(row["task_id"]),
                "split": split,
            },
        }

    return task, sft


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for row in rows:
            encoded = (
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            ).encode("utf-8")
            handle.write(encoded)
            digest.update(encoded)
            count += 1
    return {
        "path": path.name,
        "rows": count,
        "sha256": digest.hexdigest(),
    }


def _dataset_revision(dataset_id: str, revision: str) -> str | None:
    try:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(dataset_id, revision=revision)
        return info.sha
    except Exception:
        return None


def build_dataset(
    *,
    output_dir: Path,
    revision: str,
    train_public_tests: int,
    eval_public_tests: int,
    limit: int | None = None,
) -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "dataset dependencies are missing; install with "
            "pip install -e '.[data]'"
        ) from exc

    dataset = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        revision=revision,
    )

    artifacts: dict[str, Any] = {}
    split_fingerprints: dict[str, str | None] = {}

    for split in ("train", "validation", "test"):
        rows = dataset[split]
        split_fingerprints[split] = getattr(rows, "_fingerprint", None)
        if limit is not None:
            rows = rows.select(range(min(limit, len(rows))))

        tasks: list[dict[str, Any]] = []
        sft_rows: list[dict[str, Any]] = []
        public_count = (
            train_public_tests if split == "train" else eval_public_tests
        )

        for row in rows:
            task, sft = build_mbpp_task(
                dict(row),
                split=split,
                public_test_count=public_count,
            )
            tasks.append(task)
            if sft is not None:
                sft_rows.append(sft)

        artifacts[f"{split}_tasks"] = _write_jsonl(
            output_dir / f"{split}_tasks.jsonl",
            tasks,
        )
        if split == "train":
            artifacts["train_sft"] = _write_jsonl(
                output_dir / "train_sft.jsonl",
                sft_rows,
            )

    source_sha = _dataset_revision(DATASET_ID, revision)
    manifest = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "requested_revision": revision,
        "resolved_dataset_sha": source_sha,
        "expected_split_sizes": EXPECTED_SPLIT_SIZES,
        "split_fingerprints": split_fingerprints,
        "train_public_tests": train_public_tests,
        "eval_public_tests": eval_public_tests,
        "limit": limit,
        "artifacts": artifacts,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reproducible MBPP artifacts for CodeRL-Lab"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--train-public-tests", type=int, default=2)
    parser.add_argument("--eval-public-tests", type=int, default=1)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-split limit for smoke tests only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_dataset(
        output_dir=args.output_dir,
        revision=args.revision,
        train_public_tests=args.train_public_tests,
        eval_public_tests=args.eval_public_tests,
        limit=args.limit,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
