"""TRAIN-017 7B QLoRA continuation of frozen TRAIN-011 SFT using actual GRPO.

The reward function invokes the Docker executable-code sandbox, never a
language-model mock/reward approximation. Heldout oracles are verified for
identity but never loaded into the GRPO training Dataset or reward function.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import yaml

from coderl_lab.datasets.train017_new_algorithms import digest
MODEL = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
FROZEN_SFT_SHA = "25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8"
FROZEN_DATA_EXPERIMENT = "TRAIN-017-7B-NEW-CODE-GRPO"
from coderl_lab.train.train014b_reward_sandbox import (
    IMAGE_ID, check_pinned_docker, reward_factory,
)

def read(path:Path)->list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def preflight(cfg:dict, root:Path, config_path:Path)->dict:
    dataset=cfg["dataset"]
    paths={
        "train":root/"train.jsonl",
        "holdout_public":root/"heldout_public.jsonl",
        "holdout_oracle":root/"heldout_oracle.jsonl",
        "manifest":root/"manifest.json",
    }
    manifest=json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if (manifest["experiment"]!=FROZEN_DATA_EXPERIMENT
        or cfg["model"]["name_or_path"]!=MODEL
        or cfg["model"]["revision"]!=REVISION
        or manifest["model_revision"]!=REVISION
        or manifest["train_tasks"]!=48
        or manifest["heldout_tasks"]!=24
        or manifest["train_heldout_spec_overlap"]!=0
        or manifest["heldout_oracle_never_seen_by_training"] is not True
        or cfg["reward"]["docker_sha256"]!=IMAGE_ID
        or cfg["grpo"].get("adapter_sha256")!=FROZEN_SFT_SHA
        or cfg["grpo"].get("qlora_base_quantization")!="nf4_double_quant_bf16"):
        raise ValueError("TRAIN014 model/holdout/reward frozen contract changed")
    for kind in ("train","holdout_public","holdout_oracle"):
        observed=digest(paths[kind])
        expected_config_key={"train":"train_sha256","holdout_public":"heldout_public_sha256","holdout_oracle":"heldout_oracle_sha256"}[kind]
        if observed!=manifest["sha256"][kind] or observed!=dataset[expected_config_key]:
            raise ValueError("TRAIN014 dataset source sha256 changed: "+kind)
    prompts=read(paths["train"])
    holdout_prompt=read(paths["holdout_public"])
    if (len(prompts)!=48 or len(holdout_prompt)!=24
        or {r["task_id"] for r in prompts} & {r["task_id"] for r in holdout_prompt}
        or len({r["task_id"] for r in prompts})!=48):
        raise ValueError("TRAIN014 training/holdout task ID overlap")
    for row in prompts:
        if (set(row)!={"task_id","prompt","test_cases","task_kind"}
            or len(row["prompt"])!=1 or row["prompt"][0]["role"]!="user"
            or "test_cases" in str(row["prompt"])):
            raise ValueError("test oracle exposed as a prompt")
    return {
        "experiment":"TRAIN-017",
        "preflight":"verified",
        "config_sha256":digest(config_path),
        "train_sha256":digest(paths["train"]),
        "holdout_public_sha256":digest(paths["holdout_public"]),
        "holdout_oracle_sha256":digest(paths["holdout_oracle"]),
        "model_revision":REVISION,
        "sandbox_docker_image_sha256":check_pinned_docker(),
        "train_tasks":len(prompts),"heldout_tasks":len(holdout_prompt),
        "heldout_seen_during_optimizer_updates":False,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--sft-adapter",type=Path,required=True)
    p.add_argument("--data-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--max-steps",type=int,required=True)
    p.add_argument("--run-kind",choices=("smoke","formal"),required=True)
    args=p.parse_args()
    config=yaml.safe_load(args.config.read_text(encoding="utf-8"))
    g=config["grpo"]
    allowed_steps=int(config["runs"][args.run_kind+"_steps"])
    if (args.max_steps!=allowed_steps or config["experiment"]["id"]!="TRAIN-017"
        or g["num_generations"]!=4
        or g["per_device_train_batch_size"]!=4
        or g["beta"]!=0.0
        or g["loss_type"]!="grpo"
        or g["use_vllm"] is not False):
        raise ValueError("run does not match fixed real GRPO training config")
    expected_output=Path(config["runs"][args.run_kind+"_output"])
    if args.output_dir!=expected_output:
        raise ValueError("must use unique frozen experimental output folder")
    pre=preflight(config,args.data_dir,args.config)
    source_adapter=args.sft_adapter/"adapter_model.safetensors"
    if not source_adapter.is_file() or digest(source_adapter)!=FROZEN_SFT_SHA:
        raise ValueError("TRAIN-016 must start from the exact TRAIN-011 7B SFT Adapter SHA")
    pre["initial_sft_adapter_sha256"]=FROZEN_SFT_SHA
    pre["warm_started_from_train011_7b_sft"]=True
    args.output_dir.mkdir(parents=True,exist_ok=True)
    identity={
        **pre,
        "run_kind":args.run_kind,
        "optimizer_steps":args.max_steps,
        "source_file_sha256":digest(Path(__file__)),
        "reward_source_sha256":digest(Path(__file__).parent/"train014b_reward_sandbox.py"),
        "execution":"real_GPU_TRL_GRPOTrainer_with_docker_verifiable_code_reward",
    }
    marker=args.output_dir/"identity.json"
    if marker.exists():
        raise FileExistsError("TRAIN014 refuses to overwrite or resume a partly run identity")
    marker.write_text(json.dumps(identity,indent=2)+"\n")
    import torch
    import peft,transformers,trl
    from datasets import Dataset
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    from trl import GRPOConfig,GRPOTrainer

    if not torch.cuda.is_available():raise RuntimeError("TRAIN014 requires real CUDA")
    random.seed(config["experiment"]["seed"])
    torch.manual_seed(config["experiment"]["seed"])
    torch.cuda.manual_seed_all(config["experiment"]["seed"])
    torch.cuda.reset_peak_memory_stats()
    tokenizer=AutoTokenizer.from_pretrained(
        MODEL,revision=REVISION,use_fast=True,local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    quant=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    backbone=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=REVISION,local_files_only=True,
        dtype=torch.bfloat16,device_map={"":0},
        quantization_config=quant,
    )
    backbone.config.use_cache=False
    model=PeftModel.from_pretrained(
        backbone,str(args.sft_adapter),adapter_name="default",is_trainable=True,
    )
    if not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("TRAIN017 must use frozen 7B QLoRA, not full BF16 backbone")
    initial_state=load_file(str(source_adapter),device="cpu")
    def assert_initial_sft_exact(peft_model):
        state=get_peft_model_state_dict(peft_model)
        if set(state)!=set(initial_state):
            raise ValueError("GRPO SFT initial LoRA adapter tensor keys differ")
        if any(not torch.equal(state[k].detach().to("cpu").to(initial_state[k].dtype),initial_state[k])
               for k in initial_state):
            raise ValueError("SFT LoRA model parameter values changed before optimizer")
    assert_initial_sft_exact(model)
    # This contains only the 36 TRAIN task/oracle rows. No heldout file content.
    train_rows=read(args.data_dir/"train.jsonl")
    # Apache Arrow cannot mix list-valued and string-valued x in a nested struct.
    # Keep the frozen tests byte-equivalent by encoding only this metadata
    # as JSON strings; the reward callback decodes before execution.
    for record in train_rows:
        record["test_cases"]=json.dumps(record["test_cases"],sort_keys=True)
    dataset=Dataset.from_list(train_rows)
    training=GRPOConfig(
        output_dir=str(args.output_dir),
        seed=int(config["experiment"]["seed"]),
        data_seed=int(config["experiment"]["seed"]),
        max_steps=args.max_steps,
        per_device_train_batch_size=int(g["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(g["gradient_accumulation_steps"]),
        num_generations=int(g["num_generations"]),
        learning_rate=float(g["learning_rate"]),
        warmup_steps=int(g["warmup_steps"]),
        max_completion_length=int(g["max_completion_length"]),
        temperature=float(g["temperature"]),
        top_p=float(g["top_p"]),
        beta=float(g["beta"]),
        loss_type=g["loss_type"],
        num_iterations=int(g["num_iterations"]),
        scale_rewards="group",
        use_vllm=False,
        bf16=True,fp16=False,
        max_grad_norm=float(g["max_grad_norm"]),
        gradient_checkpointing=bool(g["gradient_checkpointing"]),
        logging_steps=1,logging_first_step=True,
        save_strategy="no",report_to="none",
        disable_dropout=True,
        log_completions=False,
        dataloader_num_workers=0,
    )
    grader=reward_factory(args.output_dir/"reward_attempts.jsonl")
    started=time.monotonic()
    trainer=GRPOTrainer(
        model=model,
        reward_funcs=grader,
        args=training,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=None,
    )
    # TRL 1.14.1 converts quantized-model trainable LoRA to BF16
    # during trainer initialization. Restore FP32 BEFORE optimizer creation,
    # consistent with TRAIN-013 numeric precision control.
    assert_initial_sft_exact(trainer.model)
    trainable_params=[(n,p) for n,p in trainer.model.named_parameters() if p.requires_grad]
    if (not trainable_params
        or any(".lora_" not in n or ".default." not in n for n,p in trainable_params)):
        raise RuntimeError("unexpected trainable parameters outside 7B LoRA adapter")
    dtype_before=sorted({str(p.dtype) for _,p in trainable_params})
    for n,p in trainable_params:
        p.data=p.data.float()
    trainable=sorted({str(p.dtype) for _,p in trainable_params})
    if trainable!=["torch.float32"]:
        raise RuntimeError("TRAIN016 trainable LoRA not truly FP32 after TRL initialization")
    trained_count=sum(p.numel() for _,p in trainable_params)
    if trained_count!=2523136:
        raise RuntimeError(f"wrong SFT LoRA size (expected2523136 got{trained_count})")
    trainer_result=trainer.train()
    runtime=time.monotonic()-started
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    adapter_file=args.output_dir/"adapter_model.safetensors"
    if not adapter_file.exists():
        raise RuntimeError("GRPOTrainer did not save real LoRA adapter")
    attempts=read(args.output_dir/"reward_attempts.jsonl")
    if not attempts:
        raise RuntimeError("no real Docker reward evaluations occurred")
    if len(attempts)!=args.max_steps*int(g["per_device_train_batch_size"]):
        # Some GRPO generation schedulers may redraw samples across steps:
        # still capture the true count; never claim the expected budget.
        print(f"WARNING: reward count differs from simple estimate: {len(attempts)}",flush=True)
    history=list(trainer.state.log_history)
    (args.output_dir/"log_history.json").write_text(
        json.dumps(history,ensure_ascii=False,indent=2,default=str)+"\n"
    )
    counts={}
    for event in attempts:
        key=str(event["reward"])
        counts[key]=counts.get(key,0)+1
    grouping={}
    for event in attempts:
        grouping.setdefault(event["task_id"],[]).append(event["reward"])
    varying=sum(len(set(rewards))>1 for rewards in grouping.values())
    final_state=get_peft_model_state_dict(trainer.model)
    changed_elements=sum(int((final_state[k].detach().cpu().to(initial_state[k].dtype)!=initial_state[k]).sum().item())
                         for k in initial_state)
    result={
        **identity,
        "status":"real_GPU_GRPO_completed",
        "global_step":int(trainer.state.global_step),
        "trainer_max_steps":int(trainer.state.max_steps),
        "trainable_parameters":trained_count,
        "trainable_adapter_dtype_before_post_trainer_cast":dtype_before,
        "initial_SFT_parameter_elements_changed_after_grpo":changed_elements,
        "warm_start_source_sft_adapter_sha256":FROZEN_SFT_SHA,
        "backbone_quantization":"NF4_double_quant_BF16",
        "trainable_lora_dtype":sorted(trainable),
        "source_prompt_count":len(dataset),
        "docker_reward_calls":len(attempts),
        "reward_histogram":counts,
        "reward_group_variation_tasks":varying,
        "full_correct_reward_count":sum(float(x["reward"])==1. for x in attempts),
        "mean_reward":sum(float(x["reward"]) for x in attempts)/len(attempts),
        "actual_train_metrics":dict(trainer_result.metrics),
        "saved_adapter_sha256":digest(adapter_file),
        "reward_attempt_log_sha256":digest(args.output_dir/"reward_attempts.jsonl"),
        "gpu_peak_allocated_mib":round(torch.cuda.max_memory_allocated()/2**20,2),
        "gpu_peak_reserved_mib":round(torch.cuda.max_memory_reserved()/2**20,2),
        "train_runtime_seconds":round(runtime,2),
        "versions":{"torch":torch.__version__,"trl":trl.__version__,
                    "transformers":transformers.__version__,"peft":peft.__version__},
        "holds_training_oracle_only":True,
        "heldout_code_pass_rates_not_yet_measured":True,
        "interpretive_warning":"A positive reward/GRPO step is not proof of improvement; all-equal group rewards mean no group advantage gradient.",
    }
    (args.output_dir/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
