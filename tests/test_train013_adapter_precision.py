"""TRAIN-013 LoRA param dtype after actual TRL DPOTrainer initialization."""
import pytest

from coderl_lab.train.dpo import enforce_trainable_lora_dtype


def test_bad_precision_does_not_even_try_loading_torch():
    with pytest.raises(ValueError, match="unsupported"):
        enforce_trainable_lora_dtype(None, "f16_wildcard")


def fake_model():
    torch = pytest.importorskip("torch")
    from torch import nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Module()
            self.layer.lora_A = nn.ModuleDict({
                "default": nn.Linear(3, 2, bias=False, dtype=torch.bfloat16),
                "ref": nn.Linear(3, 2, bias=False, dtype=torch.bfloat16),
            })
            self.layer.lora_B = nn.ModuleDict({
                "default": nn.Linear(2, 3, bias=False, dtype=torch.bfloat16),
                "ref": nn.Linear(2, 3, bias=False, dtype=torch.bfloat16),
            })
            for part in ("lora_A", "lora_B"):
                pair = getattr(self.layer, part)
                pair["ref"].weight.data.copy_(pair["default"].weight.data)
                pair["ref"].weight.requires_grad_(False)
    return Tiny()


def test_bf16_and_fp32_casts_preserve_frozen_reference():
    torch=pytest.importorskip("torch")
    for mode in ("bf16", "fp32"):
        model=fake_model()
        before=[p.clone() for n,p in model.named_parameters() if ".ref." in n]
        x=enforce_trainable_lora_dtype(model, mode)
        assert x["trainable_tensors"]==2
        assert x["frozen_reference_tensors"]==2
        assert x["paired_ref_initially_identical"] is True
        dtype=torch.bfloat16 if mode=="bf16" else torch.float32
        assert all(p.dtype==dtype for n,p in model.named_parameters() if p.requires_grad)
        assert all(torch.equal(p, old) and not p.requires_grad for old, (n,p) in zip(
            before, [(n,p) for n,p in model.named_parameters() if ".ref." in n]
        ))


def test_reject_different_or_mutable_initial_reference():
    torch=pytest.importorskip("torch")
    model=fake_model()
    model.layer.lora_A.ref.weight.data.add_(1)
    with pytest.raises(ValueError, match="initial policy differ"):
        enforce_trainable_lora_dtype(model,"fp32")
    model=fake_model()
    model.layer.lora_A.ref.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="reference"):
        enforce_trainable_lora_dtype(model,"fp32")
