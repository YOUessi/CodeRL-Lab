"""TRAIN-009A: real online, token-level PPO-Clip + value baseline + RM + KL.

This code intentionally implements its own small auditable PPO loop because
the pinned TRL 1.14.1 has no importable PPOTrainer. Unlike GRPO/RLOO, this
*has an explicit trainable state-value head* and generalized advantage
estimation. It is a short CUDA engineering smoke, not a formal claim that
PPO or the reward model improves human preferences.

Source identities, policy adapter, reward adapter, and prompt-selection
protocol must be frozen BEFORE any GPU sampling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from coderl_lab.train.ppo_data import freeze_online_ppo_prompts, sha_file
from coderl_lab.train.ppo_math import (
    generalized_advantages,
    normalize_advantages,
    terminal_kl_rewards,
)
from coderl_lab.train.reward_model import pack_reward_tokens


SFT_SHA = "d8846aa5fb5fd6958d30a24611efd9fb99eb91469a60192937646b58557ce12d"
REWARD_SHA = "586fb4cb21bb3408507d6e179d1bd858aca35d7c3e46b1dbef84234d3adbd28c"
MODEL = "Qwen/Qwen3-1.7B-Base"
MODEL_REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"


def load_config(path: Path) -> dict[str, Any]:
    data=yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data,dict) or data.get("experiment",{}).get("id")!="TRAIN-009A":
        raise ValueError("wrong PPO-Clip smoke config")
    return data


def frozen_provenance(
    *,
    config_path: Path,
    data_root: Path,
    sft_dir: Path,
    reward_dir: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    cfg=load_config(config_path)
    model=cfg["model"]
    policy=cfg["policy"]
    rew=cfg["reward"]
    data=cfg["data"]
    roll=cfg["rollout"]
    algo=cfg["ppo"]
    if any((
        model["name_or_path"]!=MODEL,
        model["revision"]!=MODEL_REVISION,
        model["quantization"]!="nf4",
        not model["double_quant"],
        policy["expected_adapter_sha256"]!=SFT_SHA,
        rew["expected_adapter_sha256"]!=REWARD_SHA,
        not data["exclude_reward_training_prompts"],
        not data["exclude_all_sft_and_reward_heldout_prompts"],
        int(algo["max_steps"])!=4,
        int(roll["prompts_per_step"])!=2,
        int(roll["rollouts_per_prompt"])!=2,
        int(data["n_rollout_prompts"])!=8,
        int(data["n_probe_prompts"])!=4,
        int(roll["max_new_tokens"])<1,
        int(roll["policy_prompt_max_tokens"])<8,
        int(algo["epochs_per_batch"])<1,
        not 0<float(algo["clip_epsilon"])<1,
        not 0<float(roll["sampling_top_p"])<=1,
        float(roll["sampling_temperature"])<=0,
        cfg["output"]["no_training_on_validation_labels"] is not True,
    )):
        raise ValueError("one or more frozen PPO smoke dimensions changed")
    sft_summary=json.loads((sft_dir/"run_summary.json").read_text(encoding="utf-8"))
    rm_summary=json.loads((reward_dir/"run_summary.json").read_text(encoding="utf-8"))
    if any((
        sha_file(sft_dir/"adapter_model.safetensors")!=SFT_SHA,
        sha_file(reward_dir/"adapter_model.safetensors")!=REWARD_SHA,
        sft_summary.get("saved_adapter_sha256")!=SFT_SHA,
        sft_summary.get("num_examples")!=4096,
        sft_summary.get("model")!=MODEL,
        sft_summary.get("requested_model_revision")!=MODEL_REVISION,
        sft_summary.get("quantization",{}).get("mode")!="nf4",
        rm_summary.get("status")!="formal",
        rm_summary.get("experiment")!="TRAIN-008A",
        rm_summary.get("train_pairs")!=2048,
        rm_summary.get("heldout_pairs")!=256,
        rm_summary.get("optimizer_steps")!=256,
        rm_summary.get("adapter_sha256")!=REWARD_SHA,
        rm_summary.get("training_identity",{}).get("formal") is not True,
        rm_summary.get("training_identity",{}).get("model_revision")!=MODEL_REVISION,
    )):
        raise ValueError("SFT/reward model weights or run summaries are not frozen")
    n=int(data["n_rollout_prompts"])
    probe=int(data["n_probe_prompts"])
    prompts,source=freeze_online_ppo_prompts(
        data_root=data_root,n=n+probe,seed=int(roll["seed"]))
    training=prompts[:n]
    holdout_probe=prompts[n:]
    ids={x["prompt_sha256"] for x in training}
    if len(ids)!=n or ids & {x["prompt_sha256"] for x in holdout_probe}:
        raise ValueError("PPO train/probe prompt leakage")
    provenance={
        "experiment":"TRAIN-009A",
        "kind":"real_on_policy_PPO-Clip_with_value_head_4step_smoke",
        "config_sha256":sha_file(config_path),
        "sft_adapter_sha256":SFT_SHA,
        "reward_adapter_sha256":REWARD_SHA,
        "sft_training_summary_sha256":sha_file(sft_dir/"run_summary.json"),
        "reward_training_summary_sha256":sha_file(reward_dir/"run_summary.json"),
        "frozen_data":source,
        "train_prompt_sha256":[p["prompt_sha256"] for p in training],
        "probe_prompt_sha256":[p["prompt_sha256"] for p in holdout_probe],
        "no_private_or_hidden_tests":True,
        "no_heldout_preference_labels_used_for_policy_gradient":True,
        "is_formal_benchmark":False,
        "seed":int(roll["seed"]),
    }
    return training,holdout_probe,provenance


def _activate(model, name: str) -> None:
    if name not in ("policy","reference"):
        raise ValueError("must select PPO policy or frozen SFT reference")
    model.set_adapter(name)
    # PEFT.set_adapter() can change parameter requires_grad flags; ensure the
    # reference never receives gradients even when active.
    for parameter_name, parameter in model.named_parameters():
        if ".reference." in parameter_name:
            parameter.requires_grad_(False)
        elif ".policy." in parameter_name:
            parameter.requires_grad_(name=="policy")


def _token_logprobs_and_values(model, critic, full_ids, prompt_length, *, device, hidden: bool):
    import torch
    import torch.nn.functional as F
    if not full_ids or not 1<=prompt_length<len(full_ids):
        raise ValueError("PPO log-prob scoring needs prompt plus sampled response")
    token_ids=torch.tensor([full_ids],dtype=torch.long,device=device)
    outputs=model(input_ids=token_ids,output_hidden_states=hidden,use_cache=False)
    # Index prompt_length-1 predicts first response token.
    logits=outputs.logits[0,prompt_length-1:-1,:].float()
    actions=token_ids[0,prompt_length:]
    if logits.shape[0]!=actions.shape[0]:
        raise AssertionError("state/action logit shift mismatch")
    log_probs=F.log_softmax(logits,dim=-1).gather(1,actions[:,None]).squeeze(-1)
    if not hidden:
        return log_probs,None
    hidden_states=outputs.hidden_states[-1][0,prompt_length-1:-1,:].float()
    values=critic(hidden_states).squeeze(-1)
    if values.shape!=log_probs.shape:
        raise AssertionError("PPO value/action shape mismatch")
    return log_probs,values


def _make_rollout(model, critic, ref_reward_model, tok, *, prompt: dict[str,str],
                  config:dict[str,Any], device) -> dict[str,Any]:
    import torch
    p_cfg=config["rollout"]
    r_cfg=config["reward"]
    a_cfg=config["ppo"]
    prompt_ids=tok.encode(prompt["prompt"],add_special_tokens=False)
    prompt_ids=prompt_ids[-int(p_cfg["policy_prompt_max_tokens"]):]
    if not prompt_ids:
        raise ValueError("sampled empty PPO prompt")
    _activate(model,"policy")
    model.eval()
    critic.eval()
    with torch.inference_mode():
        result=model.generate(
            input_ids=torch.tensor([prompt_ids],dtype=torch.long,device=device),
            do_sample=True,
            temperature=float(p_cfg["sampling_temperature"]),
            top_p=float(p_cfg["sampling_top_p"]),
            max_new_tokens=int(p_cfg["max_new_tokens"]),
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            use_cache=True,
        )
        all_ids=result[0].tolist()
        response_ids=all_ids[len(prompt_ids):]
        if not response_ids:
            raise RuntimeError("generation returned no sampled actions")
        old_logits, old_v=_token_logprobs_and_values(
            model,critic,all_ids,len(prompt_ids),device=device,hidden=True)
        _activate(model,"reference")
        ref_logits,_=_token_logprobs_and_values(
            model,critic,all_ids,len(prompt_ids),device=device,hidden=False)
        _activate(model,"policy")
        old=old_logits.float().cpu().tolist()
        ref=ref_logits.float().cpu().tolist()
        values=old_v.float().cpu().tolist()
        completion=tok.decode(response_ids,skip_special_tokens=True).strip()
        if completion:
            reward_ids,_=pack_reward_tokens(
                tok,prompt=prompt["prompt"],completion=completion,
                max_length=int(r_cfg["max_length"]),
                max_prompt_tokens=int(r_cfg["max_prompt_tokens"]),
                min_response_tokens=int(r_cfg["min_response_tokens"]),
            )
            x=torch.tensor([reward_ids],dtype=torch.long,device=device)
            reward=float(ref_reward_model(
                input_ids=x,attention_mask=torch.ones_like(x)
            ).logits.float().reshape(-1)[0].item())
        else:
            reward=float(r_cfg["empty_generation_reward"])
    rewards=terminal_kl_rewards(
        old_logprobs=old,reference_logprobs=ref,scalar_reward=reward,
        kl_coefficient=float(a_cfg["kl_coefficient"]),
        reward_scale=float(r_cfg["scalar_scale"]),
        reward_cap=float(r_cfg["scalar_clip_abs"]),
    )
    adv,returns=generalized_advantages(
        rewards,values,gamma=float(a_cfg["gamma"]),
        gae_lambda=float(a_cfg["gae_lambda"]),
    )
    if len(response_ids)!=len(rewards):
        raise AssertionError("PPO rewards must map to each generated action")
    return {
        "prompt_sha256":prompt["prompt_sha256"],
        "full_ids":all_ids,
        "prompt_length":len(prompt_ids),
        "old_logprobs":old,
        "reference_logprobs":ref,
        "advantages":adv,
        "returns":returns,
        "reward_score_raw":reward,
        "response_tokens":len(response_ids),
        "sampled_kl_mean":mean(o-r for o,r in zip(old,ref,strict=True)),
        "empty_response":not bool(completion),
    }


def _ppo_minibatch_step(model, critic, optimizer, rollout:dict[str,Any],
                        cfg:dict[str,Any], normalized_adv:list[float],
                        *,device) -> dict[str,float]:
    import torch
    _activate(model,"policy")
    model.eval()       # Disable stochastic LoRA dropout in PPO probability ratios.
    critic.train()
    new_logprobs,values=_token_logprobs_and_values(
        model,critic,rollout["full_ids"],rollout["prompt_length"],
        device=device,hidden=True,
    )
    old=torch.tensor(rollout["old_logprobs"],device=device,dtype=torch.float32)
    advantage=torch.tensor(normalized_adv,device=device,dtype=torch.float32)
    target=torch.tensor(rollout["returns"],device=device,dtype=torch.float32)
    if not (new_logprobs.shape==old.shape==advantage.shape==values.shape==target.shape):
        raise AssertionError("PPO new/old/advantage/value target token sizes disagree")
    clip=float(cfg["ppo"]["clip_epsilon"])
    ratio=torch.exp(torch.clamp(new_logprobs-old,-20,20))
    objective=torch.minimum(ratio*advantage,torch.clamp(ratio,1-clip,1+clip)*advantage)
    actor_loss=-objective.mean()
    critic_loss=0.5*torch.mean((values.float()-target)**2)
    total=actor_loss+float(cfg["ppo"]["value_loss_coefficient"])*critic_loss
    if not torch.isfinite(total).item():
        raise FloatingPointError("PPO actor/critic update became nonfinite")
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    grads=[p for g in optimizer.param_groups for p in g["params"]]
    grad_norm=torch.nn.utils.clip_grad_norm_(
        grads,float(cfg["ppo"]["gradient_clip_norm"]))
    if not torch.isfinite(grad_norm).item():
        raise FloatingPointError("PPO gradient norm not finite")
    optimizer.step()
    return {
        "policy_loss":float(actor_loss.detach().item()),
        "critic_loss":float(critic_loss.detach().item()),
        "objective":float(total.detach().item()),
        "clip_fraction":float(((ratio-1).abs()>clip).float().mean().detach().item()),
        "ratio_mean":float(ratio.mean().detach().item()),
        "grad_norm":float(grad_norm.detach().item()),
    }


def _artifact_adapter(root:Path,model)->tuple[Path,str]:
    root.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(str(root),selected_adapters=["policy"],safe_serialization=True)
    models=list(root.rglob("adapter_model.safetensors"))
    if len(models)!=1:
        raise RuntimeError("expected exactly one policy-only PEFT safetensors file")
    return models[0],sha_file(models[0])


def train_online_ppo(
    *,
    config_path:Path, data_root:Path, sft_dir:Path,
    reward_dir:Path, output_dir:Path | None=None,
) -> dict[str,Any]:
    import torch
    from torch import nn
    from transformers import AutoModelForCausalLM,AutoModelForSequenceClassification,AutoTokenizer,BitsAndBytesConfig
    from peft import PeftModel, prepare_model_for_kbit_training
    from safetensors.torch import save_file
    if not torch.cuda.is_available():
        raise RuntimeError("TRAIN-009A requires a real CUDA device")
    cfg=load_config(config_path)
    train,probe,provenance=frozen_provenance(
        config_path=config_path,data_root=data_root,
        sft_dir=sft_dir,reward_dir=reward_dir,
    )
    root=output_dir or Path(cfg["output"]["dir"])
    if root.exists():
        raise FileExistsError("PPO smoke output already exists; no implicit restart/overwrite")
    root.mkdir(parents=True,exist_ok=False)
    (root/"training_identity.json").write_text(
        json.dumps(provenance,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    seed=int(cfg["rollout"]["seed"])
    random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    device=torch.device("cuda",torch.cuda.current_device())
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_REVISION,trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token=tok.eos_token
    quant=BitsAndBytesConfig(
        load_in_4bit=True,bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=MODEL_REVISION,dtype=torch.bfloat16,
        trust_remote_code=True,quantization_config=quant,
        device_map={"":torch.cuda.current_device()},
    )
    base.config.use_cache=False
    base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=False)
    model=PeftModel.from_pretrained(
        base,str(sft_dir),adapter_name="policy",is_trainable=True)
    model.load_adapter(str(sft_dir),adapter_name="reference",is_trainable=False)
    _activate(model,"policy")
    model.eval()

    rm_base=AutoModelForSequenceClassification.from_pretrained(
        MODEL,revision=MODEL_REVISION,num_labels=1,dtype=torch.bfloat16,
        trust_remote_code=True,quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16,
        ),device_map={"":torch.cuda.current_device()},
    )
    rm_base.config.pad_token_id=tok.pad_token_id
    rm_base.config.use_cache=False
    reward_model=PeftModel.from_pretrained(rm_base,str(reward_dir),is_trainable=False)
    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad_(False)

    hidden_size=int(base.config.hidden_size)
    critic=nn.Linear(hidden_size,1,dtype=torch.float32,device=device)
    nn.init.zeros_(critic.weight)
    nn.init.zeros_(critic.bias)
    policy_params=[
        p for name,p in model.named_parameters() if ".policy." in name
    ]
    if not policy_params:
        raise RuntimeError("no trainable LoRA policy parameters")
    if any(".reference." in name and p.requires_grad for name,p in model.named_parameters()):
        raise AssertionError("PPO frozen reference adapter became trainable")
    optimizer=torch.optim.AdamW([
        {"params":policy_params,"lr":float(cfg["ppo"]["policy_learning_rate"])},
        {"params":list(critic.parameters()),"lr":float(cfg["ppo"]["value_learning_rate"])},
    ],weight_decay=0.0)

    # Immutable rollout selection (no reward/policy outcome-dependent choices).
    p_cfg=cfg["rollout"]
    a_cfg=cfg["ppo"]
    steps=int(a_cfg["max_steps"])
    prompts_per_step=int(p_cfg["prompts_per_step"])
    copies=int(p_cfg["rollouts_per_prompt"])
    if len(train)!=steps*prompts_per_step:
        raise AssertionError("rollout prompt budget must match fixed optimizer steps")
    t0=time.perf_counter()
    log_path=root/"update_history.jsonl"
    initial_ref_max_difference=None
    stats=[]
    for step in range(steps):
        work=train[step*prompts_per_step:(step+1)*prompts_per_step]
        episodes=[]
        for prompt in work:
            for _ in range(copies):
                episode=_make_rollout(
                    model,critic,reward_model,tok,prompt=prompt,
                    config=cfg,device=device,
                )
                difference=max(abs(a-b) for a,b in zip(
                    episode["old_logprobs"],episode["reference_logprobs"],strict=True))
                if step==0:
                    initial_ref_max_difference=(
                        difference if initial_ref_max_difference is None
                        else max(initial_ref_max_difference,difference)
                    )
                    if difference>0.02:
                        raise RuntimeError("SFT policy and frozen SFT reference did not start identical")
                episodes.append(episode)
        advantages=[v for e in episodes for v in e["advantages"]]
        normalized=normalize_advantages(
            advantages,epsilon=float(a_cfg["advantage_normalization_epsilon"]))
        offset=0
        update_stats=[]
        for e in episodes:
            count=len(e["advantages"])
            e["normalized_advantages"]=normalized[offset:offset+count]
            offset+=count
        for epoch in range(int(a_cfg["epochs_per_batch"])):
            for e in episodes:
                update_stats.append(_ppo_minibatch_step(
                    model,critic,optimizer,e,cfg,e["normalized_advantages"],
                    device=device,
                ))
        report={
            "optimizer_update_index":step+1,
            "episodes":len(episodes),
            "tokens":sum(e["response_tokens"] for e in episodes),
            "unique_train_prompts":len(work),
            "mean_reward_model_score":mean(e["reward_score_raw"] for e in episodes),
            "mean_old_policy_sampled_KL_to_SFT_reference":mean(e["sampled_kl_mean"] for e in episodes),
            "mean_policy_loss":mean(x["policy_loss"] for x in update_stats),
            "mean_value_loss":mean(x["critic_loss"] for x in update_stats),
            "mean_clipped_fraction":mean(x["clip_fraction"] for x in update_stats),
            "max_grad_norm":max(x["grad_norm"] for x in update_stats),
            "empty_generations":sum(bool(e["empty_response"]) for e in episodes),
            "optimizer_minibatch_updates":len(update_stats),
            "selected_prompt_sha256":[p["prompt_sha256"] for p in work],
        }
        stats.append(report)
        with log_path.open("a",encoding="utf-8") as out:
            out.write(json.dumps(report,ensure_ascii=False)+"\n")
        print(
            f"TRAIN-009A PPO-Clip step {step+1}/{steps}, "
            f"episodes {len(episodes)}, rm={report['mean_reward_model_score']:.4f}, "
            f"sample_kl={report['mean_old_policy_sampled_KL_to_SFT_reference']:.4f}",
            flush=True,
        )

    # Persist actual adapter, value head and optimizer for engineering audit.
    policy_adapter,adapter_sha=_artifact_adapter(root/"policy_adapter",model)
    save_file(
        {name:tensor.detach().cpu().contiguous()
         for name,tensor in critic.state_dict().items()},
        root/"value_head.safetensors",
    )
    torch.save({
        "optimizer":optimizer.state_dict(),
        "torch_cpu_rng":torch.get_rng_state(),
        "torch_cuda_rng":torch.cuda.get_rng_state_all(),
        "completed_step":steps,
        "source_identity":provenance,
    },root/"optimizer_state.pt")
    duration=time.perf_counter()-t0
    summary={
        "experiment":"TRAIN-009A",
        "status":"real_cuda_ppo_clip_smoke_not_formal",
        "method":"token-level PPO-Clip with trainable scalar critic and GAE",
        "model":MODEL,
        "model_revision":MODEL_REVISION,
        "steps":steps,
        "unique_train_prompts":len(train),
        "heldout_probe_prompts_reserved_not_trained":len(probe),
        "training_episodes":sum(x["episodes"] for x in stats),
        "generated_tokens":sum(x["tokens"] for x in stats),
        "actual_optimizer_minibatch_steps":sum(x["optimizer_minibatch_updates"] for x in stats),
        "initial_SFT_reference_logprob_max_abs_difference":initial_ref_max_difference,
        "mean_raw_reward_model_score":mean(x["mean_reward_model_score"] for x in stats),
        "mean_sampled_on_policy_KL_to_SFT_reference":mean(x["mean_old_policy_sampled_KL_to_SFT_reference"] for x in stats),
        "mean_PPO_clipped_fraction":mean(x["mean_clipped_fraction"] for x in stats),
        "policy_adapter_sha256":adapter_sha,
        "policy_adapter_path":str(policy_adapter),
        "value_head_sha256":sha_file(root/"value_head.safetensors"),
        "optimizer_state_sha256":sha_file(root/"optimizer_state.pt"),
        "source_identity":provenance,
        "elapsed_seconds_wall":duration,
        "gpu":{
            "name":torch.cuda.get_device_name(0),
            "peak_reserved_bytes":torch.cuda.max_memory_reserved(),
        },
        "claim_limits":[
            "Only four small on-policy PPO updates, not a benchmark-quality online RLHF run.",
            "The frozen reward model is a learned imperfect proxy; optimizing it can reward-hack without human quality gains.",
            "No heldout preference labels, LCB private cases or MBPP hidden tests used during optimization.",
            "PPO reward score movement is not independent test accuracy; generalization requires new data and a preregistered baseline.",
            "The reserved probe prompts are not used for gradient updates; probe quality testing is separately scoped.",
        ],
    }
    (root/"run_summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return summary


def main() -> None:
    p=argparse.ArgumentParser(description="TRAIN-009A real on-policy PPO-Clip engineering smoke")
    p.add_argument("--config",type=Path,default=Path("configs/train009a_ppo_clip_rlhf_smoke.yaml"))
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--sft-adapter",type=Path,required=True)
    p.add_argument("--reward-adapter",type=Path,required=True)
    p.add_argument("--output-dir",type=Path)
    a=p.parse_args()
    train_online_ppo(
        config_path=a.config,data_root=a.data_root,
        sft_dir=a.sft_adapter,reward_dir=a.reward_adapter,
        output_dir=a.output_dir,
    )


if __name__=="__main__":
    main()
