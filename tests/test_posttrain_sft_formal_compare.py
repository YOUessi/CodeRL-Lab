from __future__ import annotations

import hashlib
import json
from pathlib import Path
import yaml
import pytest

from coderl_lab.analysis.posttrain_sft_formal_compare import (
    compare_formal_runs, last_eval_loss, actual_optimizer_steps,
)


def _files(tmp_path: Path) -> dict:
    fixed_data = {"source": "H4", "revision": "locked", "train_sha": "t", "validation_sha": "v"}
    fixed_model = {"name_or_path": "Qwen/Qwen3-1.7B-Base", "revision": "pinned", "dtype": "bfloat16"}
    lora = {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["q_proj", "v_proj"]}
    training = {"learning_rate": 1e-4, "seed": 42, "num_train_epochs": 1, "gradient_accumulation_steps": 16}
    manifest = {
        "stage": "sft", "track": "A-posttraining",
        "full_scan": True, "prompt_overlap": 0,
        "train": {"rows":4096,"sha256":"TRAINHASH"},
        "validation":{"rows":256,"sha256":"EVALHASH"},
    }
    manifest_path = tmp_path/"manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    dirs = []
    configs = []
    for quant_mode, loss, eval_loss, peak in (
        ("nf4",1.14,1.13,4000000000),
        ("none",1.09,1.12,6000000000),
    ):
        d = tmp_path/quant_mode
        d.mkdir()
        adapter_content = f"adapter {quant_mode}".encode()
        (d/"adapter_model.safetensors").write_bytes(adapter_content)
        digest = hashlib.sha256(adapter_content).hexdigest()
        cfg = {
            "model":fixed_model,
            "data":fixed_data,
            "lora":lora,
            "quantization":{"mode":quant_mode,"double_quant":True},
            "training":{**training,"output_dir":str(d)},
        }
        cp = tmp_path/f"{quant_mode}.yaml"
        cp.write_text(yaml.safe_dump(cfg))
        summary = {
            "model":fixed_model["name_or_path"],
            "requested_model_revision":fixed_model["revision"],
            "quantization":{"mode":quant_mode},
            "num_examples":4096,
            "num_heldout_validation_examples":256,
            "saved_adapter_sha256":digest,
            "train_data_sha256":"TRAINHASH",
            "heldout_data_sha256":"EVALHASH",
            "cuda_available":True,
            "seed":42,
            "dtype":"bfloat16",
            "lora":lora,
            "metrics":{"train_loss":loss},
            "elapsed_seconds_wall":1200 if quant_mode=="nf4" else 1400,
            "gpu":{"peak_reserved_bytes":peak},
        }
        (d/"run_summary.json").write_text(json.dumps(summary))
        (d/"log_history.json").write_text(json.dumps([
            {"step":1,"loss":loss+0.1},
            {"step":252,"loss":loss},
            {"step":252,"eval_loss":eval_loss},
        ]))
        dirs.append(d)
        configs.append(cp)
    return dict(nf4_config_path=configs[0],bf16_config_path=configs[1],
                manifest_path=manifest_path,nf4_dir=dirs[0],bf16_dir=dirs[1])


def test_real_artifact_comparison_requires_and_reports_matched_inputs(tmp_path: Path):
    paths = _files(tmp_path)
    result = compare_formal_runs(**paths, gpu_contention="observed")
    assert result["formal_data"] is True
    assert result["gpu_contention_during_comparison"] == "observed"
    assert result["wallclock_performance_is_clean_hardware_comparison"] is False
    assert result["nf4_qlora"]["optimizer_steps"] == 252
    assert result["bf16_lora"]["eval_loss"] == 1.12
    assert result["relative_memory_saving_nf4"] == pytest.approx(1-4/6)
    assert result["differences_bf16_minus_nf4"]["heldout_eval_loss"] == pytest.approx(-0.01)


def test_changed_weights_rejected_even_when_summary_claims_success(tmp_path: Path):
    paths = _files(tmp_path)
    (paths["bf16_dir"]/"adapter_model.safetensors").write_text("tampered")
    with pytest.raises(ValueError, match="provenance/adapter"):
        compare_formal_runs(**paths)


def test_changed_model_or_training_parameters_rejected(tmp_path: Path):
    paths = _files(tmp_path)
    p = paths["bf16_config_path"]
    obj = yaml.safe_load(p.read_text())
    obj["training"]["learning_rate"] = 2e-4
    p.write_text(yaml.safe_dump(obj))
    with pytest.raises(ValueError, match="hyperparameters"):
        compare_formal_runs(**paths)


def test_not_enough_effective_optimizer_steps_rejected(tmp_path: Path):
    paths = _files(tmp_path)
    p = paths["bf16_dir"]/"log_history.json"
    history=json.loads(p.read_text())
    history[1]["step"] = 251
    history[2]["step"] = 251
    p.write_text(json.dumps(history))
    with pytest.raises(ValueError, match="optimizer step counts"):
        compare_formal_runs(**paths)


def test_log_must_contain_finite_heldout_eval_loss():
    with pytest.raises(ValueError, match="missing/nonfinite"):
        last_eval_loss([{"step": 1, "loss": 1.2}])
    with pytest.raises(ValueError, match="nonfinite"):
        last_eval_loss([{"eval_loss": float("nan")}])
    with pytest.raises(ValueError, match="optimizer steps"):
        actual_optimizer_steps([{"loss":0.1}])
