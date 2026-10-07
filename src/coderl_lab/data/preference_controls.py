from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_reverse(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        new = dict(row)
        new["chosen"] = str(row["rejected"])
        new["rejected"] = str(row["chosen"])
        new["control_mode"] = "reverse"
        out.append(new)
    return out


def build_noop(rows: list[dict[str, Any]], *, side: str = "chosen") -> list[dict[str, Any]]:
    if side not in {"chosen", "rejected"}:
        raise ValueError("side must be chosen or rejected")
    out: list[dict[str, Any]] = []
    for row in rows:
        new = dict(row)
        value = str(row[side])
        new["chosen"] = value
        new["rejected"] = value
        new["control_mode"] = f"noop_{side}"
        out.append(new)
    return out


def write_control(
    *,
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    mode: str,
    noop_side: str,
) -> dict[str, Any]:
    rows = load_jsonl(input_path)
    if mode == "reverse":
        out = build_reverse(rows)
    elif mode == "noop":
        out = build_noop(rows, side=noop_side)
    else:
        raise ValueError("mode must be reverse or noop")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in out:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    same_prompts = all(
        str(a["prompt"]) == str(b["prompt"])
        for a, b in zip(rows, out, strict=True)
    )
    if mode == "reverse":
        pair_relation_ok = all(
            str(a["chosen"]) == str(b["rejected"])
            and str(a["rejected"]) == str(b["chosen"])
            for a, b in zip(rows, out, strict=True)
        )
    else:
        pair_relation_ok = all(
            str(b["chosen"]) == str(b["rejected"])
            for b in out
        )

    summary = {
        "mode": mode,
        "noop_side": noop_side if mode == "noop" else None,
        "pairs": len(rows),
        "same_prompts": same_prompts,
        "pair_relation_ok": pair_relation_ok,
        "source_sha256": sha256(input_path),
        "output_sha256": sha256(output_path),
        "purpose": (
            "reverse: test preference-direction dependence; "
            "noop: zero-gradient training-pipeline control"
        ),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build EXP-004C preference controls")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--mode", choices=["reverse", "noop"], required=True)
    p.add_argument("--noop-side", choices=["chosen", "rejected"], default="chosen")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = write_control(
        input_path=args.input,
        output_path=args.output,
        summary_path=args.summary,
        mode=args.mode,
        noop_side=args.noop_side,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
