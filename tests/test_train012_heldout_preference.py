import pytest
from coderl_lab.analysis.train012_heldout_preference import (
    BETA, comparison, validate_rows,
)


def row(idx, sft_margin, dpo_margin):
    return {
        "index": idx,
        "task_id_sha256": f"hash_{idx}",
        "scorable": True,
        "chosen_target_tokens": 3,
        "rejected_target_tokens": 3,
        "sft_chosen_logp": sft_margin,
        "sft_rejected_logp": 0.0,
        "dpo_chosen_logp": dpo_margin,
        "dpo_rejected_logp": 0.0,
    }


def test_fixed_holdout_pair_scoring_and_coverage():
    # One positive rescue, one regression, and a 512-context ineligible row.
    rows = [
        row(0, 1.0, 2.0), row(1, -1.0, 1.0),
        row(2, 1.0, -1.0),
        {"index": 3, "scorable": False,
         "reason": "no_completion_token_in_frozen_512_context"},
    ]
    out = comparison(rows)
    assert out["heldout_pairs_total"] == 4
    assert out["scorable_pairs"] == 3
    assert out["unscorable_512_context_pairs"] == 1
    assert out["sft_raw_chosen_preference_accuracy"] == pytest.approx(2/3)
    assert out["dpo_raw_chosen_preference_accuracy"] == pytest.approx(2/3)
    assert out["sft_to_dpo_wrong_to_correct"] == 1
    assert out["sft_to_dpo_correct_to_wrong"] == 1
    assert out["dpo_relative_to_sft_implicit_reward_margin_mean"] == pytest.approx(BETA/3)
    assert out["paired_bootstrap"]["raw_preference_accuracy_delta"]["observed"] == 0


def test_bad_holdout_data_fails_before_gpu_model_load():
    rows = [
        {
            "task_id": f"pref_{n}",
            "prompt": f"user_{n}",
            "chosen": "preferred",
            "rejected": "other",
        }
        for n in range(256)
    ]
    assert len(validate_rows(rows)) == 256
    rows[1]["task_id"] = rows[0]["task_id"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_rows(rows)
    rows[1]["task_id"] = "pref_1"
    rows[1]["rejected"] = rows[1]["chosen"]
    with pytest.raises(ValueError, match="identical"):
        validate_rows(rows)
