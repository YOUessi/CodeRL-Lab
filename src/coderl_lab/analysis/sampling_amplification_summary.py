from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ARM_RE = re.compile(r"scale(025|050|100|200)_seed(101|202|303)")
SCALE_MAP = {"025": 0.25, "050": 0.5, "100": 1.0, "200": 2.0}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def parse_arm(name: str) -> tuple[float, int]:
    match = ARM_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"invalid arm name: {name}")
    return SCALE_MAP[match.group(1)], int(match.group(2))


def summarize(
    *,
    greedy_root: Path,
    logit_root: Path,
    stochastic_summary: dict[str, Any],
) -> dict[str, Any]:
    sft_eval = _load(greedy_root / "sft_eval" / "summary.json")
    sft_greedy_pass1 = float(sft_eval["pass_at_k"]["pass@1"])

    arms: dict[str, Any] = {}
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)

    for name, stochastic_row in sorted(stochastic_summary["arms"].items()):
        scale, seed = parse_arm(name)
        arm_root = greedy_root / name
        greedy_behavior = _load(arm_root / "behavior_vs_sft.json")
        greedy_eval = _load(arm_root / "evaluation" / "summary.json")
        logit = _load(logit_root / f"{name}.json")["summary"]

        stochastic_exact = float(stochastic_row["exact_match_fraction_vs_sft"])
        greedy_exact = float(greedy_behavior["exact_completion_match_fraction"])
        stochastic_changed = 1.0 - stochastic_exact
        greedy_changed = 1.0 - greedy_exact

        row = {
            "name": name,
            "scale": scale,
            "seed": seed,
            "stochastic_exact_match_fraction": stochastic_exact,
            "greedy_exact_match_fraction": greedy_exact,
            "stochastic_changed_fraction": stochastic_changed,
            "greedy_changed_fraction": greedy_changed,
            "sampling_amplification_gap": stochastic_changed - greedy_changed,
            "sft_greedy_pass1": sft_greedy_pass1,
            "candidate_greedy_pass1": float(
                greedy_eval["pass_at_k"]["pass@1"]
            ),
            "greedy_pass1_delta_vs_sft": (
                float(greedy_eval["pass_at_k"]["pass@1"])
                - sft_greedy_pass1
            ),
            "prompt_kl_mean": float(logit["kl_ref_to_candidate"]["mean"]),
            "prompt_kl_median": float(logit["kl_ref_to_candidate"]["median"]),
            "prompt_kl_p95": float(logit["kl_ref_to_candidate"]["p95"]),
            "prompt_tv_mean": float(logit["total_variation"]["mean"]),
            "prompt_tv_p95": float(logit["total_variation"]["p95"]),
            "prompt_top1_agreement": float(logit["top1_agreement_fraction"]),
            "prompt_top5_jaccard_mean": float(logit["top5_jaccard"]["mean"]),
            "centered_logit_rms_mean": float(
                logit["centered_logit_rms_delta"]["mean"]
            ),
        }
        arms[name] = row
        grouped[scale].append(row)

    by_scale: dict[str, Any] = {}
    for scale, rows in sorted(grouped.items()):
        by_scale[str(scale)] = {
            "scale": scale,
            "num_seeds": len(rows),
            "stochastic_changed_fraction": _stats(
                [float(row["stochastic_changed_fraction"]) for row in rows]
            ),
            "greedy_changed_fraction": _stats(
                [float(row["greedy_changed_fraction"]) for row in rows]
            ),
            "sampling_amplification_gap": _stats(
                [float(row["sampling_amplification_gap"]) for row in rows]
            ),
            "greedy_pass1_delta_vs_sft": _stats(
                [float(row["greedy_pass1_delta_vs_sft"]) for row in rows]
            ),
            "prompt_kl_mean": _stats(
                [float(row["prompt_kl_mean"]) for row in rows]
            ),
            "prompt_tv_mean": _stats(
                [float(row["prompt_tv_mean"]) for row in rows]
            ),
            "prompt_top1_agreement": _stats(
                [float(row["prompt_top1_agreement"]) for row in rows]
            ),
            "prompt_top5_jaccard_mean": _stats(
                [float(row["prompt_top5_jaccard_mean"]) for row in rows]
            ),
        }

    return {
        "sft_greedy_pass1": sft_greedy_pass1,
        "arms": arms,
        "by_scale": by_scale,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--greedy-root", type=Path, required=True)
    parser.add_argument("--logit-root", type=Path, required=True)
    parser.add_argument("--stochastic-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = summarize(
        greedy_root=args.greedy_root,
        logit_root=args.logit_root,
        stochastic_summary=_load(args.stochastic_summary),
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
