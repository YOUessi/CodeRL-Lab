from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _unresolved_count(diag: dict[str, Any], name: str) -> int:
    return sum(
        int(count)
        for key, count in diag.get("top_unresolved_names", [])
        if str(key) == name
    )


def _runtime_failure_count(diag: dict[str, Any], name: str) -> int:
    return sum(
        int(count)
        for key, count in diag.get("runtime_failure_types", [])
        if str(key) == name
    )


def compare(
    *,
    sft_eval: dict[str, Any],
    dpo_eval: dict[str, Any],
    sft_diag: dict[str, Any],
    dpo_diag: dict[str, Any],
    passk_bootstrap: dict[str, Any],
) -> dict[str, Any]:
    validation = {
        "sft": {
            "pass_at_k": sft_eval["pass_at_k"],
            "hidden_mean_pass_rate": sft_eval["hidden_mean_pass_rate"],
            "solved_tasks": sft_eval["solved_tasks"],
            "all_samples_correct_tasks": sft_eval["all_samples_correct_tasks"],
            "syntax_failures": sft_eval["syntax_failures"],
        },
        "dpo": {
            "pass_at_k": dpo_eval["pass_at_k"],
            "hidden_mean_pass_rate": dpo_eval["hidden_mean_pass_rate"],
            "solved_tasks": dpo_eval["solved_tasks"],
            "all_samples_correct_tasks": dpo_eval["all_samples_correct_tasks"],
            "syntax_failures": dpo_eval["syntax_failures"],
        },
        "delta": {
            "pass@1": dpo_eval["pass_at_k"]["pass@1"] - sft_eval["pass_at_k"]["pass@1"],
            "pass@4": dpo_eval["pass_at_k"]["pass@4"] - sft_eval["pass_at_k"]["pass@4"],
            "pass@8": dpo_eval["pass_at_k"]["pass@8"] - sft_eval["pass_at_k"]["pass@8"],
            "pass@16": dpo_eval["pass_at_k"]["pass@16"] - sft_eval["pass_at_k"]["pass@16"],
            "hidden_mean_pass_rate": (
                dpo_eval["hidden_mean_pass_rate"]
                - sft_eval["hidden_mean_pass_rate"]
            ),
            "solved_tasks": dpo_eval["solved_tasks"] - sft_eval["solved_tasks"],
            "syntax_failures": dpo_eval["syntax_failures"] - sft_eval["syntax_failures"],
        },
    }

    diagnostics = {
        "sft": {
            "dependency_incomplete_count": sft_diag["dependency_incomplete_count"],
            "runtime_unclean_count": sft_diag["runtime_unclean_count"],
            "name_error_count": _runtime_failure_count(sft_diag, "NameError"),
            "unresolved_re": _unresolved_count(sft_diag, "re"),
            "unresolved_math": _unresolved_count(sft_diag, "math"),
        },
        "dpo": {
            "dependency_incomplete_count": dpo_diag["dependency_incomplete_count"],
            "runtime_unclean_count": dpo_diag["runtime_unclean_count"],
            "name_error_count": _runtime_failure_count(dpo_diag, "NameError"),
            "unresolved_re": _unresolved_count(dpo_diag, "re"),
            "unresolved_math": _unresolved_count(dpo_diag, "math"),
        },
    }
    diagnostics["delta"] = {
        key: diagnostics["dpo"][key] - diagnostics["sft"][key]
        for key in diagnostics["sft"]
    }

    return {
        "validation": validation,
        "diagnostics": diagnostics,
        "passk_bootstrap": passk_bootstrap,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare SFT vs repair DPO")
    p.add_argument("--sft-eval", type=Path, required=True)
    p.add_argument("--dpo-eval", type=Path, required=True)
    p.add_argument("--sft-diagnostics", type=Path, required=True)
    p.add_argument("--dpo-diagnostics", type=Path, required=True)
    p.add_argument("--passk-bootstrap", type=Path, required=True)
    p.add_argument("--output", type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = compare(
        sft_eval=_load(args.sft_eval),
        dpo_eval=_load(args.dpo_eval),
        sft_diag=_load(args.sft_diagnostics),
        dpo_diag=_load(args.dpo_diagnostics),
        passk_bootstrap=_load(args.passk_bootstrap),
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
