"""EXP-004M: distribution-transfer audit from immutable EXP-004K/L summaries.

A *post-hoc descriptive diagnostic*. This deliberately never reads hidden
test cases, raw completions, or per-task code and never adjusts the frozen
EXP-004J/K/L gate or intervention.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

Z95 = 1.959963984540054


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def wilson_interval(successes: int, total: int, z: float = Z95) -> list[float]:
    """95% Wilson score interval for an observed Bernoulli proportion."""
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("invalid successes/total")
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided conditional hypergeometric test (no SciPy dependency).

    Sample sizes in the low hundreds; sums in log-space and math.fsum.
    Used only as a *descriptive post-hoc* check; not preregistered evidence.
    """
    if min(a, b, c, d) < 0:
        raise ValueError("table cells must be nonnegative")
    row1, row2, successes = a + b, c + d, a + c
    n = row1 + row2
    if n == 0:
        raise ValueError("empty contingency table")

    def log_choose(n_: int, k: int) -> float:
        if k < 0 or k > n_:
            return float("-inf")
        return math.lgamma(n_ + 1) - math.lgamma(k + 1) - math.lgamma(n_ - k + 1)

    def pmf(x: int) -> float:
        return math.exp(
            log_choose(successes, x)
            + log_choose(n - successes, row1 - x)
            - log_choose(n, row1)
        )

    observed = pmf(a)
    low = max(0, row1 - (n - successes))
    high = min(row1, successes)
    # Standard Fisher two-sided: sum of tables not more probable than observed.
    return min(
        1.0,
        math.fsum(pmf(x) for x in range(low, high + 1)
                  if pmf(x) <= observed * (1 + 1e-9)),
    )


def read_frozen_counts(
    k: dict[str, Any], l: dict[str, Any]
) -> tuple[dict[str, int], dict[str, int]]:
    if k.get("experiment") != "EXP-004K" or k.get("tasks") != 500:
        raise ValueError("requires frozen EXP-004K 500-task summary")
    if l.get("tasks") != 175 or l.get("claim_limit") != "formal 175-task external-distribution replication":
        raise ValueError("requires EXP-004L 175-task official summary")
    if l.get("private_tests_accessed_only_after_freeze") is not True:
        raise ValueError("EXP-004L isolation attestation missing")
    if l.get("frozen_inputs", {}).get("private_tests_accessed") is not False:
        raise ValueError("EXP-004L gate leaked hidden tests")
    if l.get("frozen_inputs", {}).get("official_evaluator_commit") != "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24":
        raise ValueError("official LCB checker revision mismatch")

    kb = _positive_int(k["counts"]["baseline_hidden_correct"], "K baseline")
    lb = _positive_int(l["conditions"]["baseline"]["correct_tasks"], "L baseline")
    kt = _positive_int(k["counts"]["gate_triggered_tasks"], "K gated")
    lt = _positive_int(l["gate_triggered_tasks"], "L gated")
    kw = _positive_int(k["actual_gate_audit"]["hidden_wrong_selected"], "K wrong selected")
    lw = _positive_int(l["posthoc_gate_audit"]["gate_selected_hidden_wrong"], "L wrong selected")
    ktrans = k["gate_triggered_subset"]
    ltrans = l["gate_triggered_subset"]["low_transitions"]
    kresc = _positive_int(ktrans["wrong_to_correct"], "K rescue")
    kharm = _positive_int(ktrans["correct_to_wrong"], "K harm")
    lresc = _positive_int(ltrans["wrong_to_correct"], "L rescue")
    lharm = _positive_int(ltrans["correct_to_wrong"], "L harm")
    if not (0 <= kb <= 500 and 0 <= lb <= 175 and 0 <= kw <= kt <= 500 and 0 <= lw <= lt <= 175):
        raise ValueError("counts contradict number of tasks")
    if kresc + kharm > kt or lresc + lharm > lt:
        raise ValueError("transition counts exceed gated subsets")
    if kresc > kw or lresc > lw or kharm > kt - kw or lharm > lt - lw:
        raise ValueError("correctness transitions incompatible with gate precision")

    if k["primary"]["wrong_to_correct"] != kresc or k["primary"]["correct_to_wrong"] != kharm:
        raise ValueError("K primary and gate subset transitions disagree")
    expected_l = l["conditions"]["gated_low_destabilize"]["transitions_vs_baseline"]
    if expected_l["wrong_to_correct"] != lresc or expected_l["correct_to_wrong"] != lharm:
        raise ValueError("L primary and gate subset transitions disagree")
    if kb + kresc - kharm != k["counts"]["gated_low_hidden_correct"]:
        raise ValueError("K total hidden correctness does not reconcile")
    if lb + lresc - lharm != l["conditions"]["gated_low_destabilize"]["correct_tasks"]:
        raise ValueError("L total hidden correctness does not reconcile")

    ka = {
        "tasks": 500, "base_correct": kb, "eligible": k["counts"]["eligible_tasks"],
        "gated": kt, "wrong_gated": kw, "rescued": kresc, "harmed": kharm,
        "public_pass": k["public_pass_protection"]["public_pass_tasks"],
    }
    la = {
        "tasks": 175, "base_correct": lb, "eligible": l["eligible_tasks"],
        "gated": lt, "wrong_gated": lw, "rescued": lresc, "harmed": lharm,
        "public_pass": l["public_pass_tasks"],
    }
    for label, obj in (("K", ka), ("L", la)):
        if not (0 <= obj["eligible"] <= obj["tasks"]
                and 0 <= obj["gated"] <= obj["eligible"]
                and obj["public_pass"] <= obj["tasks"]):
            raise ValueError(f"{label} eligibility/public pass totals invalid")
    return ka, la


def summarize_transfer(k: dict[str, Any], l: dict[str, Any]) -> dict[str, Any]:
    ka, la = read_frozen_counts(k, l)

    def summary(src: dict[str, int]) -> dict[str, Any]:
        n, gt, wrong, rescue = (
            src["tasks"], src["gated"], src["wrong_gated"], src["rescued"]
        )
        out: dict[str, Any] = {
            **src,
            "base_correct_fraction": src["base_correct"] / n,
            "eligible_fraction": src["eligible"] / n,
            "gate_trigger_fraction": gt / n,
            "public_pass_fraction": src["public_pass"] / n,
            "gate_precision_for_wrong": wrong / gt if gt else None,
            "gate_precision_wilson95": wilson_interval(wrong, gt) if gt else None,
            "rescue_per_trigger": rescue / gt if gt else None,
            "rescue_per_wrong_trigger": rescue / wrong if wrong else None,
            "rescue_per_trigger_wilson95": wilson_interval(rescue, gt) if gt else None,
            "rescue_per_wrong_trigger_wilson95": (
                wilson_interval(rescue, wrong) if wrong else None
            ),
            "net_correct_gain_fraction": (rescue - src["harmed"]) / n,
        }
        return out

    # The primary direct comparison on success-generation opportunity conditions
    # is rescue among *actually wrong triggered* tasks; true positives differ.
    # This is an exploratory between-benchmark comparison (non-randomized domain).
    rescue_fisher = fisher_exact_two_sided(
        ka["rescued"], ka["wrong_gated"] - ka["rescued"],
        la["rescued"], la["wrong_gated"] - la["rescued"],
    )
    ksum, lsum = summary(ka), summary(la)
    return {
        "experiment": "EXP-004M",
        "status": "posthoc_descriptive_transfer_audit",
        "source_experiments": ["EXP-004K", "EXP-004L"],
        "source_frozen_and_unmodified": True,
        "metrics": {
            "mbpp_test_500": ksum,
            "livecodebench_v6_175": lsum,
        },
        "exploratory_between_domain_comparison": {
            "rescue_per_wrong_trigger_k_minus_l": (
                ksum["rescue_per_wrong_trigger"] - lsum["rescue_per_wrong_trigger"]
            ),
            "fisher_exact_two_sided_unadjusted": rescue_fisher,
            "note": (
                "Exploratory unadjusted descriptive check. Different task difficulty,"
                " prompt formats and checker semantics are confounded; no preregistered"
                " causal or statistical superiority claim across two task datasets."
            ),
        },
        "evidence_boundaries": [
            "No token traces in the frozen Git summaries: causal pathway and "
            "candidate reachability cannot be separated yet.",
            "Gate precision is correctness detection, not rescue efficacy.",
            "The LiveCodeBench hard subset is 0/80 correct before and after intervention.",
            "The lack of gate advantage on L does not prove gating is unnecessary "
            "in other domains.",
            "Frozen L thresholds and outcomes were not changed during this diagnostic.",
        ],
        "next_experiment": (
            "Pre-register independent model-capability and intervention-efficacy "
            "controls on a genuinely unseen evaluation slice; do not tune on LCB v6."
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    k = summary["metrics"]["mbpp_test_500"]
    l = summary["metrics"]["livecodebench_v6_175"]
    def percent(value: float) -> str:
        return f"{value * 100:.2f}%"
    return (
        "# EXP-004M：从同家族复制到跨分布迁移的失效诊断\n\n"
        "> 后验描述性分析，不是新模型实验、正式独立假设检验或阈值搜索。\n\n"
        "| 指标 | EXP-004K · MBPP 500 | EXP-004L · LCB v6 175 |\n"
        "|---|---:|---:|\n"
        f"| Base 正确率 | {percent(k['base_correct_fraction'])} | {percent(l['base_correct_fraction'])} |\n"
        f"| Gate 触发率 | {percent(k['gate_trigger_fraction'])} | {percent(l['gate_trigger_fraction'])} |\n"
        f"| Gate 选中错误精度 | {percent(k['gate_precision_for_wrong'])} | {percent(l['gate_precision_for_wrong'])} |\n"
        f"| 被选中的错误题 | {k['wrong_gated']} | {l['wrong_gated']} |\n"
        f"| 错→对 (rescue) | {k['rescued']} | {l['rescued']} |\n"
        f"| 对→错 (harm) | {k['harmed']} | {l['harmed']} |\n"
        f"| rescue / 错误门控题 | {percent(k['rescue_per_wrong_trigger'])} | {percent(l['rescue_per_wrong_trigger'])} |\n"
        f"| 正式净增益 | {percent(k['net_correct_gain_fraction'])} | {percent(l['net_correct_gain_fraction'])} |\n\n"
        f"探索性两个数据集 rescue 条件比率比较（Fisher 双侧，未多重校正）："
        f" {summary['exploratory_between_domain_comparison']['fisher_exact_two_sided_unadjusted']:.6g}。"
        " 两套题不是随机分配的同一分布实验，因此它只能提示差异，不能证明归因。\n\n"
        "## 严格结论\n\n"
        "Gate 错题选择能力在 LiveCodeBench 未表现为明显退化，但被选择的错误题"
        "通过局部扰动成功变为正确的比例明显更低。"
        "这不等于我们已经证明难度地板是唯一原因，"
        "因为模型能力、任务输入格式、测试方法与解码预算同时变化。\n\n"
        "下一步先建立独立的数据/模型/干预能力对照，禁止调整冻结的 LCB v6 参数。\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mbpp-k", type=Path, required=True)
    parser.add_argument("--lcb-l", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize_transfer(
        json.loads(args.mbpp_k.read_text(encoding="utf-8")),
        json.loads(args.lcb_l.read_text(encoding="utf-8")),
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.write_text(
        render_markdown(summary),
        encoding="utf-8",
    )
    print(json.dumps({
        "K rescue_per_wrong_trigger": summary["metrics"]["mbpp_test_500"]["rescue_per_wrong_trigger"],
        "L rescue_per_wrong_trigger": summary["metrics"]["livecodebench_v6_175"]["rescue_per_wrong_trigger"],
        "descriptive_ood_only": True,
    }, indent=2))


if __name__ == "__main__":
    main()
