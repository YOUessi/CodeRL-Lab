"""TRAIN-009A reward-hacking probe: four frozen *untrained* prompts.

No policy gradients, no preference labels, no test-time prompt selection.
This is a tiny same-reward-model diagnostic, NOT evidence of human quality.
Compares greedy reference SFT and saved 4step PPO under identical token cap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from coderl_lab.train.ppo_data import sha_file
from coderl_lab.train.ppo_online import (
    MODEL,MODEL_REVISION,SFT_SHA,REWARD_SHA,
    _token_logprobs_and_values,
    frozen_provenance,
    load_config,
)
from coderl_lab.train.reward_model import pack_reward_tokens


def _sha_text(value:str)->str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def summarize_probe(rows:list[dict[str,Any]])->dict[str,Any]:
    if len(rows)!=4 or len({str(r["prompt_sha256"]) for r in rows})!=4:
        raise ValueError("must use exactly four distinct frozen probes")
    required=(
        "reference_score","ppo_score","reference_tokens","ppo_tokens",
        "reference_answer_sha256","ppo_answer_sha256",
    )
    for r in rows:
        if any(k not in r for k in required):
            raise ValueError("incomplete frozen probe row")
        if not all(math.isfinite(float(r[k])) for k in ("reference_score","ppo_score")):
            raise ValueError("nonfinite reward-model probe score")
    changed=sum(r["reference_answer_sha256"]!=r["ppo_answer_sha256"] for r in rows)
    gaps=[r["ppo_score"]-r["reference_score"] for r in rows]
    return {
        "experiment":"TRAIN-009A-probe",
        "type":"same_reward_model_fixed_greedy_untrained_prompt_smoke",
        "probe_count":4,
        "changed_greedy_outputs":changed,
        "mean_frozen_RM_score_reference":mean(r["reference_score"] for r in rows),
        "mean_frozen_RM_score_ppo":mean(r["ppo_score"] for r in rows),
        "mean_reward_proxy_score_difference":mean(gaps),
        "score_differences_per_probe":gaps,
        "mean_response_tokens_reference":mean(r["reference_tokens"] for r in rows),
        "mean_response_tokens_ppo":mean(r["ppo_tokens"] for r in rows),
        "no_human_labels_used":True,
        "no_optimizer_updates":True,
        "limits":[
            "Only four pre-frozen prompts that were not optimized by PPO; this is not an independent external generalization benchmark.",
            "The same imperfect reward proxy is used for both RL training and post-run scoring; improvements are NOT evidence of human preference.",
            "Greedy decoding may remain identical despite probability/LoRA changes; no-change does not prove models are equivalent.",
            "No UltraFeedback heldout labels, MBPP hidden tests or LCB private tests are used.",
        ],
    }


def run_probe(*,config_path:Path,data_root:Path,sft_dir:Path,
              reward_dir:Path,ppo_root:Path,output_path:Path)->dict[str,Any]:
    import torch
    from transformers import (
        AutoModelForCausalLM,AutoModelForSequenceClassification,
        AutoTokenizer,BitsAndBytesConfig,
    )
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA is required for policy/reward probe")
    cfg=load_config(config_path)
    train,probe,source=frozen_provenance(
        config_path=config_path,data_root=data_root,
        sft_dir=sft_dir,reward_dir=reward_dir,
    )
    if len(probe)!=4 or output_path.exists():
        raise FileExistsError("probe output exists or frozen probe count drifted")
    summary_path=ppo_root/"run_summary.json"
    smoke=json.loads(summary_path.read_text(encoding="utf-8"))
    if any((
        smoke.get("experiment")!="TRAIN-009A",
        smoke.get("status")!="real_cuda_ppo_clip_smoke_not_formal",
        smoke.get("steps")!=4,
        smoke.get("training_episodes")!=16,
        smoke.get("source_identity")!=source,
        smoke.get("policy_adapter_sha256")!=
            sha_file(ppo_root/"policy_adapter/policy/adapter_model.safetensors"),
    )):
        raise ValueError("PPO model/probe frozen identity mismatch")
    dev=torch.device("cuda",torch.cuda.current_device())
    tok=AutoTokenizer.from_pretrained(
        MODEL,revision=MODEL_REVISION,trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token=tok.eos_token
    q=BitsAndBytesConfig(
        load_in_4bit=True,bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=MODEL_REVISION,dtype=torch.bfloat16,
        trust_remote_code=True,quantization_config=q,
        device_map={"":torch.cuda.current_device()},
    )
    base.config.use_cache=False
    policy=PeftModel.from_pretrained(base,str(sft_dir),adapter_name="reference",is_trainable=False)
    policy.load_adapter(
        str(ppo_root/"policy_adapter/policy"),adapter_name="ppo",is_trainable=False)
    policy.eval()
    reward_base=AutoModelForSequenceClassification.from_pretrained(
        MODEL,revision=MODEL_REVISION,num_labels=1,dtype=torch.bfloat16,
        trust_remote_code=True,quantization_config=BitsAndBytesConfig(
          load_in_4bit=True,bnb_4bit_quant_type="nf4",
          bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16,
        ),device_map={"":torch.cuda.current_device()},
    )
    reward_base.config.pad_token_id=tok.pad_token_id
    reward_model=PeftModel.from_pretrained(
        reward_base,str(reward_dir),is_trainable=False)
    reward_model.eval()
    rows=[]
    for row in probe:
        prompt_ids=tok.encode(row["prompt"],add_special_tokens=False)
        prompt_ids=prompt_ids[-int(cfg["rollout"]["policy_prompt_max_tokens"]):]
        data={"prompt_sha256":row["prompt_sha256"]}
        for arm,adapter in (("reference","reference"),("ppo","ppo")):
            policy.set_adapter(adapter)
            with torch.inference_mode():
                res=policy.generate(
                    input_ids=torch.tensor([prompt_ids],device=dev,dtype=torch.long),
                    do_sample=False,
                    max_new_tokens=int(cfg["rollout"]["max_new_tokens"]),
                    eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,
                    use_cache=True,
                )[0].tolist()
                continuation_ids=res[len(prompt_ids):]
                completion=tok.decode(continuation_ids,skip_special_tokens=True).strip()
                if completion:
                    score_ids,_=pack_reward_tokens(
                        tok,prompt=row["prompt"],completion=completion,
                        max_length=int(cfg["reward"]["max_length"]),
                        max_prompt_tokens=int(cfg["reward"]["max_prompt_tokens"]),
                        min_response_tokens=int(cfg["reward"]["min_response_tokens"]),
                    )
                    t=torch.tensor([score_ids],dtype=torch.long,device=dev)
                    score=float(reward_model(
                        input_ids=t,attention_mask=torch.ones_like(t)
                    ).logits.float().reshape(-1)[0].item())
                else:
                    score=float(cfg["reward"]["empty_generation_reward"])
            data[arm+"_answer_sha256"]=_sha_text(completion)
            data[arm+"_tokens"]=len(continuation_ids)
            data[arm+"_score"]=score
        rows.append(data)
    aggregated=summarize_probe(rows)
    aggregated["frozen_inputs"]={
        "SFT_adapter_sha256":SFT_SHA,
        "reward_adapter_sha256":REWARD_SHA,
        "PPO_adapter_sha256":smoke["policy_adapter_sha256"],
        "PPO_training_identity_sha256":sha_file(ppo_root/"training_identity.json"),
        "prompt_sha256":[x["prompt_sha256"] for x in probe],
        "source_policy":"unchangedSFT-vs-4stepPPO, greedily decoded cap32",
    }
    aggregated["probe_rows"]=rows
    aggregated["gpu"]={
        "device":torch.cuda.get_device_name(0),
        "peak_reserved_bytes":torch.cuda.max_memory_reserved(),
    }
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(json.dumps(aggregated,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(aggregated,ensure_ascii=False,indent=2),flush=True)
    return aggregated


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,default=Path("configs/train009a_ppo_clip_rlhf_smoke.yaml"))
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--sft-adapter",type=Path,required=True)
    p.add_argument("--reward-adapter",type=Path,required=True)
    p.add_argument("--ppo-root",type=Path,default=Path("artifacts/posttrain-a/train009a-ppo-clip-smoke"))
    p.add_argument("--output",type=Path,default=Path("artifacts/posttrain-a/train009a-ppo-clip-smoke/probe_summary.json"))
    a=p.parse_args()
    run_probe(config_path=a.config,data_root=a.data_root,
              sft_dir=a.sft_adapter,reward_dir=a.reward_adapter,
              ppo_root=a.ppo_root,output_path=a.output)


if __name__=="__main__":
    main()
