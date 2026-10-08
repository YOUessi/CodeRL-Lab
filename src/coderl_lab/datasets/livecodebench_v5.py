"""Public-only v5 time-window control; never decode private tests here."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from coderl_lab.datasets.livecodebench import (
    DATASET_REPO, prepare_public_view, sha256_file,
)

SOURCE_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
FILENAME = "test5.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/exp004n/data"))
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.source is None:
        from huggingface_hub import hf_hub_download
        source = Path(hf_hub_download(
            repo_id=DATASET_REPO, filename=FILENAME,
            revision=SOURCE_REVISION, repo_type="dataset",
        ))
    else:
        source = args.source
    manifest = prepare_public_view(
        source_path=source, output_dir=args.output_dir,
        source_revision=SOURCE_REVISION, fine_grained_version="v5",
    )
    if manifest["fine_grained_version"] != "v5":
        raise AssertionError("wrong time-window release")
    if not manifest["tasks"] or manifest["private_tests_exported"] or manifest["private_tests_decoded"]:
        raise AssertionError("empty/unsafe public dataset")
    if manifest["source_sha256"] != sha256_file(source):
        raise AssertionError("source checksum changed")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
