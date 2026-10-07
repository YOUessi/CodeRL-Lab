import pytest

from coderl_lab.analysis.prompt_logit_sensitivity import summarize


def row(kl, tv, rms, same, jaccard, ref_margin=0.2, cand_margin=0.19):
    return {
        "kl_ref_to_candidate": kl,
        "total_variation": tv,
        "centered_logit_rms_delta": rms,
        "top1_same": same,
        "top5_jaccard": jaccard,
        "reference_top1_margin": ref_margin,
        "candidate_top1_margin": cand_margin,
    }


def test_summarize_prompt_logit_sensitivity() -> None:
    rows = [
        row(0.001, 0.01, 0.02, True, 1.0),
        row(0.003, 0.03, 0.04, False, 0.5),
        row(0.002, 0.02, 0.03, True, 0.8),
    ]
    out = summarize(rows)
    assert out["tasks"] == 3
    assert out["kl_ref_to_candidate"]["mean"] == pytest.approx(0.002)
    assert out["total_variation"]["median"] == pytest.approx(0.02)
    assert out["top1_agreement_fraction"] == pytest.approx(2 / 3)
    assert out["top5_jaccard"]["min"] == pytest.approx(0.5)
