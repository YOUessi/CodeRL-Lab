"""CPU-only quality audit of already frozen EXP-004L completions.

Retrospective diagnostics only. Never regenerates, modifies, ranks, exposes or
executes model candidate code; outputs aggregate counts without private tests.
"""
from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from coderl_lab.analysis.exp004m_local_trace_audit import sha256_file


def _syntax_ok(raw: str) -> bool:
    if not raw.strip():
        return False
    try:
        ast.parse(raw)
    except (SyntaxError, ValueError):
        return False
    return True


def summarize_quality(
    runner: dict[str, Any],
    *,
    expected_runner_sha256: str,
    actual_runner_sha256: str,
) -> dict[str, Any]:
    if expected_runner_sha256 != actual_runner_sha256:
        raise ValueError("frozen runner checksum mismatch")
    if runner.get("experiment") != "EXP-004L" or runner.get("private_tests_accessed") is not False:
        raise ValueError("invalid EXP-004L provenance/isolation")
    if runner.get("max_new_tokens") != 512:
        raise ValueError("frozen EXP-004L token cap changed")
    per_task = runner.get("per_task", {})
    if runner.get("tasks") != 175 or len(per_task) != 175:
        raise ValueError("requires complete 175-task runner")

    rows = []
    for task_id in sorted(per_task):
        task = per_task[task_id]
        baseline = str(task["baseline"]["code"])
        gated = str(task["gated_low_destabilize"]["code"])
        count = int(task["reference_token_count"])
        if count < 1 or count > 512:
            raise ValueError("invalid baseline generated token length")
        rows.append({
            "platform": str(task["platform"]),
            "difficulty": str(task["difficulty"]),
            "gated": bool(task["gate_triggered"]),
            "hit_token_cap": count >= 512,
            "length": count,
            "baseline_parseable": _syntax_ok(baseline),
            "gated_parseable": _syntax_ok(gated),
            "changed": baseline != gated,
        })

    def count_group(items: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(items)
        if not n:
            return {"tasks": 0}
        return {
            "tasks": n,
            "gate_triggered": sum(x["gated"] for x in items),
            "hit_512_token_cap": sum(x["hit_token_cap"] for x in items),
            "token_cap_fraction": sum(x["hit_token_cap"] for x in items) / n,
            "baseline_ast_parseable": sum(x["baseline_parseable"] for x in items),
            "gated_ast_parseable": sum(x["gated_parseable"] for x in items),
            "baseline_unparseable": sum(not x["baseline_parseable"] for x in items),
            "gated_unparseable": sum(not x["gated_parseable"] for x in items),
            "changed": sum(x["changed"] for x in items),
            "unparseable_at_cap": sum(
                x["hit_token_cap"] and not x["baseline_parseable"]
                for x in items
            ),
            "mean_reference_tokens": sum(x["length"] for x in items) / n,
        }

    groupings = {}
    for key in ("difficulty", "platform"):
        groupings[key] = {
            value: count_group([r for r in rows if r[key] == value])
            for value in sorted({r[key] for r in rows})
        }
    groupings["gated"] = {
        "true": count_group([r for r in rows if r["gated"]]),
        "false": count_group([r for r in rows if not r["gated"]]),
    }
    return {
        "experiment": "EXP-004M",
        "audit": "posthoc_baseline_token_cap_and_syntax",
        "frozen_runner_sha256": actual_runner_sha256,
        "overall": count_group(rows),
        "groups": groupings,
        "no_private_test_data_or_raw_code_exported": True,
        "interpretation_limit": (
            "Hitting the 512-token cap may indicate truncation, but it is not "
            "by itself proof of truncation or the causal reason for failed tests. "
            "AST parseability is necessary but not sufficient for correctness; "
            "the two task domains use different code formats."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--formal-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.formal_summary.read_text(encoding="utf-8"))
    expected = summary["frozen_inputs"]["runner_sha256"]
    output = summarize_quality(
        json.loads(args.runner.read_text(encoding="utf-8")),
        expected_runner_sha256=expected,
        actual_runner_sha256=sha256_file(args.runner),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "n": output["overall"]["tasks"],
        "cap_hit": output["overall"]["hit_512_token_cap"],
        "syntax_fail": output["overall"]["baseline_unparseable"],
        "gate_changed": output["groups"]["gated"]["true"]["changed"],
        "private_tests_exported": False,
    }, indent=2))


if __name__ == "__main__":
    main()
