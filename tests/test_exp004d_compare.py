import json
from pathlib import Path

from coderl_lab.analysis.exp004d_compare import summarize


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_summarize_three_seed_layout(tmp_path: Path) -> None:
    exp = tmp_path / "exp004d"
    sft_eval = {
        "pass_at_k": {
            "pass@1": 0.2,
            "pass@4": 0.4,
            "pass@8": 0.5,
            "pass@16": 0.6,
        },
        "solved_tasks": 6,
        "hidden_mean_pass_rate": 0.3,
    }
    sft_concentration = {
        "hhi_success_mass": 0.1,
        "effective_task_count_from_hhi": 10.0,
    }

    arm_dirs = {}
    for idx, seed in enumerate((101, 202, 303), start=1):
        name = f"matched_seed_{seed}"
        arm = exp / "eval-validation" / name
        arm_dirs[name] = arm
        write_json(
            arm / "evaluation" / "enhanced_summary.json",
            {
                "pass_at_k": {
                    "pass@1": 0.2 + 0.01 * idx,
                    "pass@4": 0.4 + 0.01 * idx,
                    "pass@8": 0.5 + 0.01 * idx,
                    "pass@16": 0.6 + 0.01 * idx,
                },
                "solved_tasks": 6 + idx,
                "hidden_mean_pass_rate": 0.3 + 0.01 * idx,
            },
        )
        analysis = exp / "analysis" / name
        write_json(
            analysis / "behavior_vs_sft.json",
            {
                "exact_completion_match_fraction": 0.7,
                "changed_rows": 10,
                "changed_tasks": 5,
                "candidate_minus_reference_diversity": {
                    "unique_fraction": 0.01,
                },
            },
        )
        write_json(
            analysis / "success_concentration.json",
            {
                "hhi_success_mass": 0.1 - 0.001 * idx,
                "effective_task_count_from_hhi": 10.0 + idx,
            },
        )
        write_json(
            analysis / "passk_vs_sft.json",
            {"pass@16": {"observed_delta": 0.01 * idx}},
        )

    out = summarize(
        sft_eval=sft_eval,
        sft_concentration=sft_concentration,
        arm_dirs=arm_dirs,
    )
    assert out["aggregate"]["num_seeds"] == 3
    assert (
        out["aggregate"]["pass_at_k_delta_vs_sft"]["pass@16"][
            "positive_seed_count"
        ]
        == 3
    )
    assert out["aggregate"]["solved_delta_vs_sft"]["mean"] == 2.0
