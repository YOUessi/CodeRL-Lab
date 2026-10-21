from pathlib import Path

import yaml


def test_train007c_v2_fixes_evaluation_memory_without_changing_training() -> None:
    formal = yaml.safe_load(Path("configs/posttrain_a_dpo_qwen3_1.7b.yaml").read_text())
    smoke = yaml.safe_load(Path("configs/posttrain_a_dpo_smoke_v2_qwen3_1.7b.yaml").read_text())
    assert formal["experiment"]["id"] == smoke["experiment"]["id"] == "TRAIN-007C"
    assert formal["model"] == smoke["model"]
    assert formal["quantization"] == smoke["quantization"]
    assert formal["initial_policy"] == smoke["initial_policy"]
    assert formal["data"] == smoke["data"]
    a, b = formal["training"], smoke["training"]
    assert {k: v for k,v in a.items() if k!="output_dir"} == {
        k: v for k,v in b.items() if k!="output_dir"}
    assert a["per_device_eval_batch_size"] == b["per_device_eval_batch_size"] == 1
    assert a["eval_accumulation_steps"] == b["eval_accumulation_steps"] == 1
    assert a["prediction_loss_only"] is b["prediction_loss_only"] is True
    assert "smoke-v2" in b["output_dir"]


def test_v2_script_keeps_original_failed_v1_artifacts_untouched() -> None:
    v1 = Path("scripts/run_posttrain_a_dpo_smoke.sh").read_text()
    v2 = Path("scripts/run_posttrain_a_dpo_smoke_v2.sh").read_text()
    assert 'OUTPUT="artifacts/posttrain-a/train007c-dpo-smoke"' in v1
    assert 'OUTPUT="artifacts/posttrain-a/train007c-dpo-smoke-v2"' in v2
    assert '--template "$TEMPLATE"' in v2
    assert 'RETRY_PRETRAINING_FAILURE' in v2
    assert '"${VERIFY_ARGS[@]}"' in v2
    assert 'QLORA_EXTRA_PYTHONPATH' in v2
