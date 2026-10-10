"""TRAIN-009A: online GRPO using a REAL, frozen NF4 reward model.

Not PPO: the installed TRL 1.14.1 exports GRPOTrainer, not PPOTrainer.
KL's reference must be the PRE-UPDATE SFT LoRA, never the bare base model.
All rewards use the trained TRAIN-008A scalar head and its packed tokens.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import yaml

from coderl_lab.datasets.train009a_online_prompts import frozen_prompt_pool
from coderl_lab.train.reward_model import pack_reward_tokens

MODEL="Qwen/Qwen3-1.7B-Base"
REVISION="ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
SFT_SHA="d8846aa5fb5fd6958d30a24611efd9fb99eb91469a60192937646b58557ce12d"
RM_SHA="586fb4cb21bb3408507d6e179d1bd858aca35d7c3e46b1dbef84234d3adbd28c"


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as handle:
        for piece in iter(lambda:handle.read(1024*1024),b""):
            h.update(piece)
    return h.hexdigest()


def validate_frozen_models(
    *,
    config:dict[str,Any],
    policy_dir:Path,
    reward_dir:Path,
)->dict[str,Any]:
    if config["experiment"]["id"]!="TRAIN-009A":
        raise ValueError("wrong experiment ID")
    if config["model"]["name_or_path"]!=MODEL or config["model"]["revision"]!=REVISION:
        raise ValueError("wrong pinned base checkpoint")
    if config["initial_policy"]["expected_sha256"]!=SFT_SHA:
        raise ValueError("invalid SFT policy pin")
    if config["reward_model"]["expected_sha256"]!=RM_SHA:
        raise ValueError("invalid learned reward pin")
    ps=policy_dir/"run_summary.json"
    rs=reward_dir/"run_summary.json"
    p=json.loads(ps.read_text(encoding="utf-8"))
    r=json.loads(rs.read_text(encoding="utf-8"))
    if any((
        p.get("model")!=MODEL,
        p.get("requested_model_revision")!=REVISION,
        p.get("num_examples")!=4096,
        p.get("quantization",{}).get("mode")!="nf4",
        p.get("saved_adapter_sha256")!=SFT_SHA,
        sha256(policy_dir/"adapter_model.safetensors")!=SFT_SHA,
        r.get("experiment")!="TRAIN-008A",
        r.get("status")!="formal",
        r.get("optimizer_steps")!=256,
        r.get("train_pairs")!=2048,
        r.get("heldout_pairs")!=256,
        r.get("training_identity",{}).get("formal") is not True,
        r.get("adapter_sha256")!=RM_SHA,
        sha256(reward_dir/"adapter_model.safetensors")!=RM_SHA,
        r.get("training_identity",{}).get("train_file_sha256")!=config["reward_model"]["source_train_sha256"],
        r.get("training_identity",{}).get("heldout_file_sha256")!=config["reward_model"]["source_heldout_sha256"],
    )):
        raise ValueError("trained SFT / Reward Model SHA or full-run provenance mismatch")
    return {
        "initial_policy_sha256":SFT_SHA,
        "frozen_reward_sha256":RM_SHA,
        "model":MODEL,
        "model_revision":REVISION,
        "reward_training_steps":r["optimizer_steps"],
        "reward_validation_accuracy":r["heldout"]["preference_accuracy"],
        "reward_source_train_sha256":r["training_identity"]["train_file_sha256"],
        "reward_source_heldout_sha256":r["training_identity"]["heldout_file_sha256"],
    }


def validate_grpo_contract(cfg:dict[str,Any],*,smoke:bool)->dict[str,Any]:
    training=cfg["training"];gen=cfg["generation"];reward=cfg["reward_model"]
    if any((
        cfg["experiment"]["id"]!="TRAIN-009A",
        cfg["model"]["quantization"]["mode"]!="nf4",
        float(training["beta"])<=0,
        training["reference_adapter"]!="ref",
        gen["num_generations"]!=4,
        training["per_device_train_batch_size"]*training["gradient_accumulation_steps"]%gen["num_generations"]!=0,
        gen["max_completion_length"]<8,
        gen["max_prompt_length"]>reward["max_prompt_tokens"],
        gen["max_prompt_length"]+gen["max_completion_length"]+1>reward["max_length"],
        reward["bounded_reward"]!="tanh",
        float(reward["bounded_reward_temperature"])<=0,
        reward["no_reward_model_parameter_updates"] is not True,
        cfg["data"]["exclude_reward_training_prompts"] is not True,
        cfg["data"]["exclude_reward_heldout_prompts"] is not True,
        cfg["data"]["exclude_sft_heldout_prompts"] is not True,
        training["scale_rewards"]!="group",
        training["loss_type"]!="grpo",
        training["mask_truncated_completions"] is not True,
    )):
        raise ValueError("invalid learned-reward on-policy GRPO contract")
    # Keep the first real smoke extremely bounded and reproducible.
    steps=1 if smoke else int(training["max_steps"])
    if not smoke and steps<1:
        raise ValueError("formal policy updates need a separately frozen positive max_steps")
    if int(training["save_steps"])<1 or int(training["save_total_limit"])<2:
        raise ValueError("online GRPO must save recoverable checkpoints")
    return {
        "reward_objective":"bounded learned frozen scalar RM, tanh(x / tau)",
        "algorithm":"GRPO (online grouped relative advantages), NOT PPO",
        "reference":"copy of trained UltraChat QLoRA SFT adapter, frozen TRL PEFT ref",
        "kl_beta":float(training["beta"]),
        "num_generations":int(gen["num_generations"]),
        "optimizer_steps":steps,
        "validation_heldout_prompts_absent_from_training":True,
    }


def assert_policy_prompt_token_budget(
    rows:list[dict[str,str]],
    tokenizer,
    *,
    max_prompt_tokens:int,
)->dict[str,Any]:
    """Fail closed rather than silently use a removed GRPOConfig argument.

    The installed TRL 1.14.1 does not accept max_prompt_length, so we must
    explicitly validate actual Qwen tokenized text before any policy rollout.
    Formal data must undergo a separate preregistered length-based selection.
    """
    if max_prompt_tokens<8 or not rows:
        raise ValueError("invalid prompt token budget or no input rows")
    sizes=[]
    for row in rows:
        p=row["prompt"]
        n=len(tokenizer(text=p)["input_ids"])
        sizes.append(n)
    over=[n for n in sizes if n>max_prompt_tokens]
    if over:
        raise ValueError(
            f"TRAIN-009A {len(over)} prompt(s) exceed explicit "
            f"{max_prompt_tokens}-token actor budget; "
            "no silent truncation or outcome-conditioned drop"
        )
    return {
        "prompts":len(rows),
        "max_length_tokens":max(sizes),
        "mean_length_tokens":statistics.mean(sizes),
        "cap_tokens":max_prompt_tokens,
        "overlong_prompts":0,
        "silently_truncated":False,
    }


class BoundedLearnedReward:
    """Dependency-injected callable: testable without CUDA or TRL.

    All outcomes logged as aggregates, never leak raw prompt/completion text.
    """
    def __init__(self, *, score_one:Callable[[str,str],float],
                 temperature:float=2.0,empty_reward:float=-1.0):
        if temperature<=0 or not math.isfinite(temperature):
            raise ValueError("invalid reward temperature")
        if not (-1<=empty_reward<=1):
            raise ValueError("empty reward must be bounded")
        self.score_one=score_one
        self.temperature=temperature
        self.empty_reward=empty_reward
        self.history:list[dict[str,Any]]=[]

    def __call__(self,prompts:list[str],completions:list[str],
                 completion_ids=None,log_metric=None,log_extra=None,**kwargs)->list[float]:
        if not prompts or len(prompts)!=len(completions):
            raise ValueError("GRPO reward receives unequal/empty prompt groups")
        raw=[];reward=[];empty=0;chars=[]
        for prompt,completion in zip(prompts,completions,strict=True):
            if not isinstance(prompt,str) or not prompt.strip():
                raise ValueError("empty online rollout prompt")
            if not isinstance(completion,str):
                raise TypeError("non-text sampled completion")
            chars.append(len(completion))
            if not completion.strip():
                empty+=1
                raw.append(None)
                reward.append(self.empty_reward)
                continue
            value=float(self.score_one(prompt,completion))
            if not math.isfinite(value):
                raise FloatingPointError("reward model emitted NaN/Inf")
            raw.append(value)
            reward.append(math.tanh(value/self.temperature))
        live=[v for v in raw if v is not None]
        audit={
            "rollouts":len(prompts),
            "empty_completions":empty,
            "raw_reward_mean":statistics.mean(live) if live else None,
            "raw_reward_min":min(live) if live else None,
            "raw_reward_max":max(live) if live else None,
            "bounded_reward_mean":statistics.mean(reward),
            "bounded_reward_std":statistics.pstdev(reward) if len(reward)>1 else 0.0,
            "nearly_saturated_fraction":sum(abs(x)>0.98 for x in reward)/len(reward),
            "mean_completion_chars":statistics.mean(chars),
            "duplicate_fraction":1-len(set(zip(prompts,completions)))/len(prompts),
        }
        self.history.append(audit)
        if log_metric is not None:
            for key in ("bounded_reward_mean","bounded_reward_std",
                        "nearly_saturated_fraction","duplicate_fraction"):
                log_metric("train009a/"+key,float(audit[key]))
        if log_extra is not None:
            log_extra("train009a_bounded_reward",reward)
        return reward

    def aggregate(self)->dict[str,Any]:
        if not self.history:
            raise ValueError("no real on-policy reward batches")
        return {
            "callback_invocations":len(self.history),
            "rewarded_rollouts":sum(row["rollouts"] for row in self.history),
            "empty_completions":sum(row["empty_completions"] for row in self.history),
            "avg_bounded_reward_mean":statistics.mean(row["bounded_reward_mean"] for row in self.history),
            "avg_bounded_reward_std":statistics.mean(row["bounded_reward_std"] for row in self.history),
            "avg_nearly_saturated_fraction":statistics.mean(row["nearly_saturated_fraction"] for row in self.history),
            "average_duplicate_fraction":statistics.mean(row["duplicate_fraction"] for row in self.history),
        }


def verify_sft_reference_clone(trainer)->dict[str,Any]:
    import torch
    model=trainer.accelerator.unwrap_model(trainer.model)
    if "default" not in model.peft_config or "ref" not in model.peft_config:
        raise RuntimeError("TRL did not create the SFT-initialized frozen ref adapter")
    parameters=dict(model.named_parameters())
    matches=0
    for name,p in parameters.items():
        if ".default." not in name:
            continue
        ref_name=name.replace(".default.",".ref.")
        if ref_name not in parameters:
            raise RuntimeError("reference PEFT parameter not found: "+ref_name)
        other=parameters[ref_name]
        if not torch.equal(p.detach(),other.detach()):
            raise RuntimeError("reference was not copied from the actual SFT adapter")
        if other.requires_grad:
            raise RuntimeError("reference LoRA incorrectly trainable")
        matches+=1
    if matches<10:
        raise RuntimeError("too few verified SFT reference adapter parameters")
    return {"reference_adapter":"ref","copied_equal_parameters":matches,
            "ref_trainable_parameters":0,"actual_pre_update_sft":True}


def run(
    *,
    config_path:Path,
    data_root:Path,
    sft_dir:Path,
    reward_dir:Path,
    output_dir:Path,
    smoke:bool,
    max_train_prompts:int|None=None,
)->dict[str,Any]:
    import torch
    import peft,trl,transformers
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,AutoModelForSequenceClassification,
        AutoTokenizer,BitsAndBytesConfig,
    )
    from peft import PeftModel,prepare_model_for_kbit_training
    from trl import GRPOTrainer,GRPOConfig

    cfg=yaml.safe_load(config_path.read_text(encoding="utf-8"))
    contract=validate_grpo_contract(cfg,smoke=smoke)
    verified=validate_frozen_models(config=cfg,policy_dir=sft_dir,reward_dir=reward_dir)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("TRAIN-009A v1 requires a single real CUDA GPU")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("online RLHF output already exists; no overwrite or invented resume")
    root=data_root
    rows,source=frozen_prompt_pool(
        root=root,seed=int(cfg["training"]["seed"]),
        max_train_prompts=max_train_prompts)
    if smoke and len(rows)!=16:
        raise ValueError("smoke must use exactly 16 disjoint online train prompts")
    if not smoke and max_train_prompts is not None:
        raise ValueError("formal online RLHF cannot cherry-pick a training subset")
    output_dir.mkdir(parents=True,exist_ok=True)
    identity={"experiment":"TRAIN-009A","smoke":smoke,
              "source_config_sha256":sha256(config_path),"trained_models":verified,
              "prompt_pool":source,"contract":contract}
    (output_dir/"training_identity.json").write_text(
        json.dumps(identity,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    seed=int(cfg["training"]["seed"])
    random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,trust_remote_code=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side="left"
    prompt_length_audit=assert_policy_prompt_token_budget(
        rows,tokenizer,max_prompt_tokens=int(cfg["generation"]["max_prompt_length"]))
    qcfg=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)

    # The frozen RM is a separate sequence classifier, not a reward heuristic.
    rtokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,trust_remote_code=True)
    if rtokenizer.pad_token_id is None: rtokenizer.pad_token=rtokenizer.eos_token
    reward_backbone=AutoModelForSequenceClassification.from_pretrained(
        MODEL,revision=REVISION,num_labels=1,dtype=torch.bfloat16,
        quantization_config=qcfg,device_map={"":torch.cuda.current_device()},
        trust_remote_code=True)
    reward_backbone.config.pad_token_id=rtokenizer.pad_token_id
    reward_model=PeftModel.from_pretrained(reward_backbone,str(reward_dir),is_trainable=False)
    reward_model.eval()
    for p in reward_model.parameters():p.requires_grad_(False)
    if any(p.requires_grad for p in reward_model.parameters()):
        raise RuntimeError("learned reward parameters were not frozen")

    rc=cfg["reward_model"]
    def score_one(prompt:str,completion:str)->float:
        ids,_=pack_reward_tokens(rtokenizer,prompt=prompt,completion=completion,
            max_length=int(rc["max_length"]),
            max_prompt_tokens=int(rc["max_prompt_tokens"]),
            min_response_tokens=int(rc["min_response_tokens"]))
        input_ids=torch.tensor([ids],device=reward_model.device,dtype=torch.long)
        with torch.inference_mode():
            value=reward_model(input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids)).logits.float().reshape(-1)[0]
        return float(value.item())

    callback=BoundedLearnedReward(
        score_one=score_one,
        temperature=float(rc["bounded_reward_temperature"]),
        empty_reward=float(rc["empty_completion_reward"]))

    # Quantized trained SFT initial policy. TRL copies its default LoRA into
    # a frozen "ref" LoRA for KL when beta>0, verified before first update.
    actor_backbone=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=REVISION,dtype=torch.bfloat16,trust_remote_code=True,
        quantization_config=qcfg,device_map={"":torch.cuda.current_device()})
    actor_backbone.config.use_cache=False
    actor_backbone=prepare_model_for_kbit_training(actor_backbone,use_gradient_checkpointing=True)
    actor=PeftModel.from_pretrained(actor_backbone,str(sft_dir),is_trainable=True)
    t=cfg["training"];g=cfg["generation"]
    args=GRPOConfig(
        output_dir=str(output_dir),
        max_steps=contract["optimizer_steps"],
        num_train_epochs=float(t["num_train_epochs"]),
        per_device_train_batch_size=int(t["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(t["gradient_accumulation_steps"]),
        learning_rate=float(t["learning_rate"]),
        weight_decay=float(t["weight_decay"]),
        lr_scheduler_type=str(t["lr_scheduler_type"]),
        logging_steps=int(t["logging_steps"]),
        save_strategy=str(t["save_strategy"]),
        save_steps=int(t["save_steps"]),save_total_limit=int(t["save_total_limit"]),
        seed=seed,data_seed=seed,
        bf16=True,gradient_checkpointing=bool(t["gradient_checkpointing"]),
        # TRL 1.14.1 has no max_prompt_length kwarg: the exact tokenizer
        # budget is checked explicitly above, before any rollout.
        max_completion_length=int(g["max_completion_length"]),
        num_generations=int(g["num_generations"]),
        temperature=float(g["temperature"]),top_p=float(g["top_p"]),
        top_k=int(g["top_k"]),beta=float(t["beta"]),
        loss_type=str(t["loss_type"]),scale_rewards=str(t["scale_rewards"]),
        mask_truncated_completions=bool(t["mask_truncated_completions"]),
        report_to="none",remove_unused_columns=False,use_vllm=False,
        log_completions=False,
    )
    trainer=GRPOTrainer(
        model=actor,reward_funcs=callback,args=args,
        train_dataset=Dataset.from_list(rows),processing_class=tokenizer)
    reference_audit=verify_sft_reference_clone(trainer)
    if trainer.accelerator.num_processes!=1:
        raise RuntimeError("GRPO v1 requires exactly one trainer process")
    if not any(p.requires_grad for p in trainer.model.parameters()):
        raise RuntimeError("policy adapter is not trainable")
    started=time.perf_counter()
    result=trainer.train()
    duration=time.perf_counter()-started
    log=list(trainer.state.log_history)
    (output_dir/"log_history.json").write_text(
        json.dumps(log,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
    callback_report=callback.aggregate()
    (output_dir/"reward_audit.json").write_text(
        json.dumps({"reward":callback_report,"history":callback.history},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    # Save ONLY the policy adapter. The copied SFT reference is frozen and
    # remains provenance, not an accidentally exported second trained model.
    trainer.model.save_pretrained(
        output_dir,selected_adapters=["default"],safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    weight=output_dir/"adapter_model.safetensors"
    if not weight.is_file():
        raise FileNotFoundError("TRAIN-009A policy adapter was not persisted")

    info={
        "experiment":"TRAIN-009A",
        "status":"smoke_only" if smoke else "formal",
        "online_algorithm":"GRPO with trained frozen scalar RM; NOT PPO",
        "global_step":trainer.state.global_step,
        "requested_optimizer_steps":contract["optimizer_steps"],
        "new_policy_adapter_sha256":sha256(weight),
        "source_sft_adapter_sha256":SFT_SHA,
        "source_reward_adapter_sha256":RM_SHA,
        "reference_policy":reference_audit,
        "reward_audit":callback_report,
        "training_identity":identity,
        "actual_policy_prompt_token_budget":prompt_length_audit,
        "metrics":dict(result.metrics),
        "gpu":{
            "name":torch.cuda.get_device_name(0),
            "peak_reserved_bytes":torch.cuda.max_memory_reserved(),
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
        },
        "versions":{"trl":trl.__version__,"peft":peft.__version__,
                    "torch":torch.__version__,"transformers":transformers.__version__},
        "elapsed_seconds_wall":duration,
        "claims_limit":"Only on-policy RM reward and KL training steps were tested; no independent human preference, code correctness or PPO gain established.",
    }
    (output_dir/"run_summary.json").write_text(
        json.dumps(info,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(info,ensure_ascii=False,indent=2),flush=True)
    return info


def main()->None:
    p=argparse.ArgumentParser(description="TRAIN-009A learned reward on-policy GRPO")
    p.add_argument("--config",type=Path,default=Path("configs/train009a_learned_reward_grpo.yaml"))
    p.add_argument("--data-root",type=Path,default=Path("data/generated/posttrain-h4-v1"))
    p.add_argument("--sft-dir",type=Path,required=True)
    p.add_argument("--reward-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--max-train-prompts",type=int)
    args=p.parse_args()
    run(config_path=args.config,data_root=args.data_root,
        sft_dir=args.sft_dir,reward_dir=args.reward_dir,output_dir=args.output_dir,
        smoke=args.smoke,max_train_prompts=args.max_train_prompts)


if __name__=="__main__":
    main()
