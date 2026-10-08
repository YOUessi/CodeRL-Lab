from __future__ import annotations

import pytest

from coderl_lab.analysis.reward_model_token_audit import audit_split


class FakeTokenizer:
    eos_token_id = 999

    def encode(self, text: str, add_special_tokens: bool = False):
        assert not add_special_tokens
        return [ord(c) for c in text]


def test_audit_keeps_chosen_and_rejected_distinguishable_when_prompt_is_long():
    rows = [
        {
            "prompt": "P" * 900,
            "chosen": "Useful explanation.",
            "rejected": "Unsupported answer.",
        },
        {
            "prompt": "Short question",
            "chosen": "A" * 600 + "correct",
            "rejected": "A" * 600 + "incorrect",
        },
    ]
    result = audit_split(
        tokenizer=FakeTokenizer(), rows=rows,
        max_length=768, max_prompt_tokens=512, min_response_tokens=128,
    )
    assert result["pairs"] == 2
    assert result["long_prompt_truncated"] == 1
    assert result["chosen_answer_truncated"] == 0
    assert result["encoded_candidates_identical"] == 0
    assert result["prompt_used_max_tokens"] == 512
    assert result["prompt_original_max_tokens"] == 900


def test_audit_reports_candidate_collisions_instead_of_hiding_them():
    result = audit_split(
        tokenizer=FakeTokenizer(),
        rows=[{
            "prompt": "P" * 900,
            "chosen": "first" + "Z" * 400,
            "rejected": "other" + "Z" * 400,
        }],
        max_length=768, max_prompt_tokens=512, min_response_tokens=128,
    )
    assert result["encoded_candidates_identical"] == 1
    assert result["chosen_answer_truncated"] == 1
    assert result["rejected_answer_truncated"] == 1


def test_audit_rejects_empty_split():
    with pytest.raises(ValueError, match="empty"):
        audit_split(
            tokenizer=FakeTokenizer(), rows=[],
            max_length=768, max_prompt_tokens=512, min_response_tokens=128,
        )
