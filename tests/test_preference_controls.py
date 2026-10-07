from coderl_lab.data.preference_controls import build_noop, build_reverse


ROWS = [
    {"prompt": "p", "chosen": "good", "rejected": "bad"},
    {"prompt": "q", "chosen": "yes", "rejected": "no"},
]


def test_reverse_swaps_all_labels() -> None:
    out = build_reverse(ROWS)
    assert out[0]["chosen"] == "bad"
    assert out[0]["rejected"] == "good"
    assert out[1]["chosen"] == "no"
    assert out[1]["rejected"] == "yes"


def test_noop_makes_identical_pair() -> None:
    out = build_noop(ROWS, side="chosen")
    assert out[0]["chosen"] == out[0]["rejected"] == "good"
    assert out[1]["chosen"] == out[1]["rejected"] == "yes"


class _FakeModule:
    def __init__(self, p=None):
        if p is not None:
            self.p = p


class _FakeModel:
    def __init__(self):
        self.mods = [
            ("layer.lora_dropout.default", _FakeModule(0.05)),
            ("layer.other_dropout", _FakeModule(0.1)),
            ("layer.lora_dropout.identity", _FakeModule()),
        ]

    def named_modules(self):
        return iter(self.mods)


def test_lora_dropout_override_only_touches_lora_dropout() -> None:
    from coderl_lab.train.dpo import override_lora_dropout

    model = _FakeModel()
    result = override_lora_dropout(model, 0.0)
    assert result["modules_changed"] == 1
    assert result["previous_values"] == [0.05]
    assert model.mods[0][1].p == 0.0
    assert model.mods[1][1].p == 0.1
