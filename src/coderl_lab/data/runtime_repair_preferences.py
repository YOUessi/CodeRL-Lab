from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from coderl_lab.execution import PythonExecutor
from coderl_lab.generation import build_prompt
from coderl_lab.schema import TestCase


MODULE_IMPORTS = {
    "math": "import math",
    "re": "import re",
    "heapq": "import heapq",
    "cmath": "import cmath",
    "collections": "import collections",
    "itertools": "import itertools",
    "functools": "import functools",
    "statistics": "import statistics",
    "bisect": "import bisect",
    "random": "import random",
    "string": "import string",
    "operator": "import operator",
    "fractions": "import fractions",
    "decimal": "import decimal",
    "sys": "import sys",
}

SYMBOL_IMPORTS = {
    # math
    "ceil": "from math import ceil",
    "floor": "from math import floor",
    "sqrt": "from math import sqrt",
    "gcd": "from math import gcd",
    "factorial": "from math import factorial",
    "comb": "from math import comb",
    "perm": "from math import perm",
    "pi": "from math import pi",
    "log": "from math import log",
    "sin": "from math import sin",
    "cos": "from math import cos",
    "tan": "from math import tan",
    # collections
    "Counter": "from collections import Counter",
    "defaultdict": "from collections import defaultdict",
    "OrderedDict": "from collections import OrderedDict",
    "deque": "from collections import deque",
    # itertools
    "groupby": "from itertools import groupby",
    "chain": "from itertools import chain",
    "tee": "from itertools import tee",
    "zip_longest": "from itertools import zip_longest",
    "product": "from itertools import product",
    "permutations": "from itertools import permutations",
    "combinations": "from itertools import combinations",
    # functools
    "reduce": "from functools import reduce",
    # heapq
    "nlargest": "from heapq import nlargest",
    "nsmallest": "from heapq import nsmallest",
    "heappush": "from heapq import heappush",
    "heappop": "from heapq import heappop",
    "heapify": "from heapq import heapify",
    # bisect
    "bisect_left": "from bisect import bisect_left",
    "bisect_right": "from bisect import bisect_right",
    # operator
    "itemgetter": "from operator import itemgetter",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def import_lines_for(unresolved_names: list[str] | tuple[str, ...]) -> list[str] | None:
    lines: list[str] = []
    for name in unresolved_names:
        line = MODULE_IMPORTS.get(str(name)) or SYMBOL_IMPORTS.get(str(name))
        if line is None:
            return None
        if line not in lines:
            lines.append(line)
    return lines


def apply_import_repair(code: str, import_lines: list[str]) -> str:
    if not import_lines:
        return code
    prefix = "\n".join(import_lines)
    return f"{prefix}\n{code.lstrip()}"


def build_verified_pairs(
    *,
    tasks_path: Path,
    predictions_path: Path,
    diagnostics_path: Path,
    output_path: Path,
    summary_path: Path,
    executor: PythonExecutor,
) -> dict[str, Any]:
    tasks = {str(x["task_id"]): x for x in load_jsonl(tasks_path)}
    predictions = {
        (str(x["task_id"]), int(x["sample_id"])): x
        for x in load_jsonl(predictions_path)
    }
    diagnostics = load_jsonl(diagnostics_path)

    pairs: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    skipped = Counter()
    import_counter = Counter()

    for diag in diagnostics:
        task_id = str(diag["task_id"])
        sample_id = int(diag["sample_id"])
        unresolved = [str(x) for x in diag.get("unresolved_names", [])]

        if not unresolved:
            skipped["no_unresolved_names"] += 1
            continue
        if float(diag.get("public_pass_rate", 0.0)) >= 1.0:
            skipped["already_public_all_pass"] += 1
            continue

        imports = import_lines_for(unresolved)
        if imports is None:
            skipped["unpatchable_name"] += 1
            continue

        pred = predictions.get((task_id, sample_id))
        task = tasks.get(task_id)
        if pred is None or task is None:
            skipped["missing_source_row"] += 1
            continue

        rejected = str(pred["completion"])
        chosen = apply_import_repair(rejected, imports)

        cases = tuple(TestCase.from_dict(x) for x in task["public_tests"])
        report = executor.run(
            chosen,
            entry_point=str(task["entry_point"]),
            cases=cases,
            setup_code=str(task.get("setup_code", "")),
        )
        if not report.syntax_ok or report.pass_rate != 1.0:
            skipped["patch_not_public_all_pass"] += 1
            continue

        prompt = build_prompt(
            str(task["prompt"]),
            str(task.get("starter_code", "")),
        )
        key = (prompt, chosen, rejected)
        if key in seen:
            skipped["duplicate_pair"] += 1
            continue
        seen.add(key)

        for line in imports:
            import_counter[line] += 1

        pairs.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "task_id": task_id,
                "sample_id": sample_id,
                "repair_type": "minimal_import",
                "unresolved_names": unresolved,
                "import_lines": imports,
                "original_public_pass_rate": float(
                    diag.get("public_pass_rate", 0.0)
                ),
                "patched_public_pass_rate": float(report.pass_rate),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in pairs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    task_ids = {str(x["task_id"]) for x in pairs}
    summary = {
        "source_predictions": len(predictions),
        "source_diagnostics": len(diagnostics),
        "verified_pairs": len(pairs),
        "distinct_tasks": len(task_ids),
        "pair_fraction_of_predictions": (
            len(pairs) / len(predictions) if predictions else 0.0
        ),
        "repair_type": "minimal_import",
        "hidden_tests_used": False,
        "import_counts": dict(import_counter.most_common()),
        "skipped": dict(skipped),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build verified minimal-import DPO preference pairs"
    )
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--diagnostics", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--memory", default="512m")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Pair construction is intentionally serial: the candidate set is small
    # after static filtering, and serial execution gives deterministic logs.
    executor = PythonExecutor(
        mode="docker",
        timeout_seconds=args.timeout,
        memory_limit=args.memory,
    )
    result = build_verified_pairs(
        tasks_path=args.tasks,
        predictions_path=args.predictions,
        diagnostics_path=args.diagnostics,
        output_path=args.output,
        summary_path=args.summary,
        executor=executor,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
