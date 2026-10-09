"""EXP-004O: purely posthoc transfer-boundary analysis from frozen summaries.

No model generation, no new private tests, no outcome-dependent tuning.
MBPP / LCB cohorts are not exchangeable. This is descriptive evidence only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> dict[str, float]:
    if not (isinstance(successes, int) and isinstance(trials, int)
            and trials > 0 and 0 <= successes <= trials):
        raise ValueError("binomial counts must be integers with 0 <= success <= total")
    p = successes / trials
    denom = 1 + z * z / trials
    center = (p + z*z/(2*trials)) / denom
    delta = z/denom*math.sqrt(p*(1-p)/trials + z*z/(4*trials*trials))
    return {"observed_fraction": p, "wilson95_low": max(0.0,center-delta),
            "wilson95_high": min(1.0,center+delta)}


def cohort(*, name: str, tasks: int, correct: int, gate_triggered: int,
           changed: int | None, rescued: int, harmed: int,
           hidden_wrong_selected: int, official_primary_success: bool) -> dict[str, Any]:
    if not (0 <= correct <= tasks and 0 <= gate_triggered <= tasks
            and 0 <= hidden_wrong_selected <= gate_triggered and
            0 <= rescued <= hidden_wrong_selected
            and 0 <= harmed <= gate_triggered-hidden_wrong_selected):
        raise ValueError("inconsistent gate/correctness counts")
    if changed is not None and not (rescued+harmed <= changed <= gate_triggered):
        raise ValueError("changed trajectories inconsistent with outcome changes")
    return {
        "cohort": name,
        "tasks": tasks,
        "baseline_correct": correct,
        "baseline_accuracy": correct/tasks,
        "gate_triggered": gate_triggered,
        "gate_trigger_rate": gate_triggered/tasks,
        "hidden_wrong_selected": hidden_wrong_selected,
        "gate_precision_posthoc": (
            hidden_wrong_selected / gate_triggered if gate_triggered else None
        ),
        "changed_trajectories": changed,
        "change_given_gate": changed/gate_triggered if changed is not None and gate_triggered else None,
        "wrong_to_correct": rescued,
        "correct_to_wrong": harmed,
        "net_correct_delta": rescued-harmed,
        "rescue_per_wrong_selected": wilson_interval(rescued, hidden_wrong_selected),
        "rescue_per_changed_trajectory": (
            wilson_interval(rescued, changed) if changed else None
        ),
        "preregistered_primary_success": official_primary_success,
    }


def build_comparison(
    k: dict[str, Any], l: dict[str, Any], m: dict[str, Any], n: dict[str, Any],
    *,
    provenance: dict[str, str] | None = None,
) -> dict[str, Any]:
    if not (k.get("experiment")=="EXP-004K" and l.get("tasks")==175
            and m.get("experiment")=="EXP-004M" and n.get("experiment")=="EXP-004N"
            and n.get("tasks")==167):
        raise ValueError("source experiment mismatch")
    if not (l.get("private_tests_accessed_only_after_freeze") is True
            and n.get("private_tests_accessed_only_after_freeze") is True
            and n.get("frozen_inputs",{}).get("private_tests_accessed") is False):
        raise ValueError("cannot audit unfrozen official evaluation")
    if m.get("runner_sha256") is not None:
        raise ValueError("wrong EXP-004M file; expected transfer summary")
    ma = m.get("l",{})
    trace = l["conditions"]["gated_low_destabilize"]["transitions_vs_baseline"]
    kn = k["actual_gate_audit"]
    ka = k["primary"]
    la = l["posthoc_gate_audit"]
    na = n["gate_audit"]
    nlow = n["arms"]["gated_low"]["transitions_vs_sft512"]
    if any((
        ka["wrong_to_correct"] != m["k"]["rescued"],
        kn["hidden_wrong_selected"] != m["k"]["hidden_wrong_selected"],
        ma["rescued"] != trace["wrong_to_correct"],
        la["gate_selected_hidden_wrong"] != ma["hidden_wrong_selected"],
        ma["low_changed_on_gated"] < ma["rescued"],
        na["gate_changed"] > na["gate_triggered"],
        n["arms"]["sft512"]["correct"] != n["arms"]["gated_low"]["correct"],
        n["contrasts"]["gated_low_minus_sft512"]["ci95_low"] != 0,
    )):
        raise ValueError("cross-source summaries are internally inconsistent")
    mbpp = cohort(
        name="MBPP-test K", tasks=k["tasks"], correct=k["counts"]["baseline_hidden_correct"],
        gate_triggered=kn["selected"], changed=None, rescued=ka["wrong_to_correct"],
        harmed=ka["correct_to_wrong"], hidden_wrong_selected=kn["hidden_wrong_selected"],
        official_primary_success=ka["preregistered_success"],
    )
    lcb6 = cohort(
        name="LiveCodeBench v6", tasks=l["tasks"],
        correct=l["conditions"]["baseline"]["correct_tasks"],
        gate_triggered=la["gate_selected_tasks"], changed=ma["low_changed_on_gated"],
        rescued=trace["wrong_to_correct"], harmed=trace["correct_to_wrong"],
        hidden_wrong_selected=la["gate_selected_hidden_wrong"],
        official_primary_success=l["primary_success"],
    )
    lcb5 = cohort(
        name="LiveCodeBench v5", tasks=n["tasks"],
        correct=n["arms"]["sft512"]["correct"],
        gate_triggered=na["gate_triggered"], changed=na["gate_changed"],
        rescued=nlow["wrong_to_correct"], harmed=nlow["correct_to_wrong"],
        hidden_wrong_selected=na["gate_selected_wrong"],
        official_primary_success=n["preregistered_primary_success"],
    )
    combined = wilson_interval(
        lcb5["wrong_to_correct"]+lcb6["wrong_to_correct"],
        lcb5["changed_trajectories"]+lcb6["changed_trajectories"],
    )
    return {
        "experiment": "EXP-004O",
        "status": "posthoc_read_only_descriptive_cross_cohort_audit",
        "new_model_generation": False,
        "new_private_test_access": False,
        "model_tuning_from_lcb_outcomes": False,
        "sources_sha256": provenance or {},
        "cohorts": {"mbpp_k":mbpp,"lcb_v6":lcb6,"lcb_v5":lcb5},
        "external_lcb_rescues_per_changed_output_exploratory": combined,
        "quality_controls": {
            "lcb_v5_base512_correct": n["arms"]["base512"]["correct"],
            "lcb_v5_sft512_correct": n["arms"]["sft512"]["correct"],
            "lcb_v5_sft1024_correct": n["arms"]["sft1024"]["correct"],
            "lcb_v5_base512_hit_token_cap": n["completion_quality"]["base512"]["cap_hit"],
            "lcb_v5_sft1024_hit_token_cap": n["completion_quality"]["sft1024"]["cap_hit"],
            "lcb_v5_sft1024_preserved_sft512_prefix": n["sft_1024_preserves_512_raw_prefix_count"],
        },
        "interpretation": (
            "On both independent LCB cohorts, changing around eighty percent "
            "of gated trajectories produced near-zero correct alternatives. "
            "This supports an output-quality/ability-floor research hypothesis, "
            "not a causal identification of any single mechanism."
        ),
        "limits": [
            "All cross-cohort comparisons are posthoc, not preregistered randomized contrasts.",
            "MBPP versus LCB differs in difficulty, prompt, evaluation and data provenance.",
            "LCB v5 is older than v6; neither is a fresh prospective holdout after this audit.",
            "Do not tune thresholds, generation budget, prompts or models against these private outcomes.",
            "Wilson bands describe binomial uncertainty; they do not correct domain confounding.",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=Path, default=Path("results/exp004k/summary.json"))
    p.add_argument("--l", type=Path, default=Path("results/exp004l/summary.json"))
    p.add_argument("--m", type=Path, default=Path("results/exp004m/transfer_summary.json"))
    p.add_argument("--n", type=Path, default=Path("results/exp004n/summary.json"))
    p.add_argument("--output", type=Path, default=Path("artifacts/exp004o/summary.json"))
    args=p.parse_args()
    paths={"k":args.k,"l":args.l,"m":args.m,"n":args.n}
    sources={key:json.loads(path.read_text(encoding="utf-8")) for key,path in paths.items()}
    result=build_comparison(**sources,provenance={key:sha256_file(path) for key,path in paths.items()})
    if args.output.exists():
        raise FileExistsError("posthoc audit already exists; refusing silent overwrite")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
