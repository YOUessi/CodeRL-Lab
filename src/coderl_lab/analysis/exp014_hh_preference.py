"""EXP-014 independent-source HH helpfulness preference comparison.

No model-training, optimizer updates or prompt edits. Inputs are the 256
previously unused (within this project) HH helpful-base chosen/rejected pairs,
from a preregistered frozen SHA selection. This script refuses to score before
the 64x3 unlabeled model generations and their hashes are already frozen.

Reports BOTH total response log probability and length-normalized log
probability, to expose length biases rather than overclaim helpfulness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from coderl_lab.datasets.exp014_hh_helpful import hash_file, MODEL, MODEL_REV
from coderl_lab.analysis.exp014_blind_generate import SHA, ARMS, PUBLIC64_SHA

PUBLIC256_SHA="6ccc68a59e47cf3646ed0df5e37a2850a74520a58a9a0d00ccdbd63cd38312f3"
PRIVATE256_SHA="7999f9e8cc9ca55b60e50b50c4c57206e01656e523c0fce7704f9c9832c588c7"
MAX_TOTAL_TOKENS=768
BETA=0.1


def bootstrap(deltas:list[float],seed:int=42,replications:int=20000)->dict:
    rng=random.Random(seed);n=len(deltas)
    if n==0:raise ValueError("cannot bootstrap 0 scored pairs")
    vals=sorted(sum(deltas[rng.randrange(n)] for _ in range(n))/n
                for _ in range(replications))
    return {
        "observed":sum(deltas)/n,
        "ci95_low":vals[int(.025*replications)],
        "ci95_high":vals[int(.975*replications)-1],
        "n_bootstrap":replications,"seed":seed,"paired_unit":"HH_preference_pair",
    }


def summarize(records:list[dict])->dict:
    valid=[r for r in records if r.get("scorable") is True]
    if not valid: raise ValueError("0 scorable prompts")
    results={
        "pairs_total":len(records),
        "pairs_scorable":len(valid),
        "pairs_unscorable":len(records)-len(valid),
        "candidate_truncation_count":{
            label:sum(int(r.get(f"{label}_truncated",False)) for r in valid)
            for label in ("chosen","rejected")
        },
        "by_model":{},
        "comparison":{},
        "warning":"Historical public HH helpfulness labels, not independent human review of generated model answers",
    }
    quantities={}
    for arm in ARMS:
        sum_margin=[r[f"{arm}_chosen_sum_logp"]-r[f"{arm}_rejected_sum_logp"] for r in valid]
        mean_margin=[r[f"{arm}_chosen_mean_logp"]-r[f"{arm}_rejected_mean_logp"] for r in valid]
        quantities[arm]={"sum":sum_margin,"mean":mean_margin}
        results["by_model"][arm]={
            "raw_sum_logp_chosen_preferred":sum(v>0 for v in sum_margin),
            "raw_sum_logp_chosen_preference_accuracy":sum(v>0 for v in sum_margin)/len(valid),
            "length_normalized_chosen_preferred":sum(v>0 for v in mean_margin),
            "length_normalized_preference_accuracy":sum(v>0 for v in mean_margin)/len(valid),
            "mean_raw_sum_logp_margin":sum(sum_margin)/len(valid),
            "mean_length_normalized_margin":sum(mean_margin)/len(valid),
        }
    for left,right in (("sft","bf16_dpo"),("sft","fp32_dpo"),("bf16_dpo","fp32_dpo")):
        group={}
        for typ in ("sum","mean"):
            a=quantities[left][typ];b=quantities[right][typ]
            group[typ]={
                "right_minus_left_logp_margin":bootstrap([v-u for u,v in zip(a,b)]),
                "right_minus_left_binary_preference_accuracy":bootstrap(
                    [int(v>0)-int(u>0) for u,v in zip(a,b)]
                ),
                "wrong_to_correct":sum(u<=0 and v>0 for u,v in zip(a,b)),
                "correct_to_wrong":sum(u>0 and v<=0 for u,v in zip(a,b)),
            }
            if typ=="sum":
                group[typ]["beta_implicit_reward_margin_right_minus_left"]=bootstrap(
                    [BETA*(v-u) for u,v in zip(a,b)]
                )
        results["comparison"][f"{right}_minus_{left}"]=group
    return results


def get_tokens(tokenizer,prompt:str,answer:str,device):
    source=prompt+answer
    enc=tokenizer(
        source,add_special_tokens=False,truncation=True,
        max_length=MAX_TOTAL_TOKENS,
        return_tensors="pt",return_offsets_mapping=True,
    )
    offsets=enc.pop("offset_mapping")[0].tolist()
    selected=[i for i,(b,e) in enumerate(offsets)
              if i>0 and e>len(prompt) and e>b]
    full_length=len(tokenizer(source,add_special_tokens=False)["input_ids"])
    return enc["input_ids"].to(device),enc["attention_mask"].to(device),selected,full_length>MAX_TOTAL_TOKENS


def load_frozen(public_path:Path,private_path:Path,manifest_path:Path,gen_summary_path:Path,gen_checkpoint_path:Path)->tuple[list[dict],list[dict],dict]:
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    generated=json.loads(gen_summary_path.read_text(encoding="utf-8"))
    # Validate the existence, count, content hashes and exact public task IDs
    # of all 192 GENERATIONS BEFORE OPENING EVEN THE PRIVATE LABEL FILE.
    if (generated.get("status")!="generated_blinded_needs_human_review"
        or generated.get("answers_generated")!=192
        or generated.get("reference_labels_never_accessed") is not True
        or generated.get("num_prompts")!=64
        or generated.get("num_outputs")!=192):
        raise ValueError("must freeze 192 model generations before reading new preference labels")
    if hash_file(gen_checkpoint_path)!=generated["generation_checkpoint_sha256"]:
        raise ValueError("generation checkpoint SHA drift before label access")
    if (manifest.get("experiment")!="EXP-014"
        or manifest["sha256"]["public_prompts"]!=PUBLIC256_SHA
        or manifest["sha256"]["private_pairs"]!=PRIVATE256_SHA
        or manifest["sha256"]["generation_prompts"]!=PUBLIC64_SHA
        or hash_file(public_path)!=PUBLIC256_SHA):
        raise ValueError("public prompt/manifest identity changed")
    pub=[json.loads(s) for s in public_path.read_text().splitlines() if s.strip()]
    generated_rows=[json.loads(s) for s in gen_checkpoint_path.read_text().splitlines() if s.strip()]
    expected={(x["task_id"],arm) for x in pub[:64] for arm in ARMS}
    keys=[(x["task_id"],x["arm"]) for x in generated_rows]
    if len(generated_rows)!=192 or len(set(keys))!=192 or set(keys)!=expected:
        raise ValueError("generation missing or duplicated a model/task before private labels")
    if any(hashlib.sha256(x["answer"].encode()).hexdigest()!=x["answer_sha256"] for x in generated_rows):
        raise ValueError("generation answer SHA mutated before opening private labels")
    # This is the FIRST access to preference label bytes in the evaluation process.
    if hash_file(private_path)!=PRIVATE256_SHA:
        raise ValueError("frozen private HH preference file changed")
    private=[json.loads(s) for s in private_path.read_text().splitlines() if s.strip()]
    if len(pub)!=256 or len(private)!=256:
        raise ValueError("need exact 256 frozen independent HH preference pairs")
    if [x["task_id"] for x in pub]!=[x["task_id"] for x in private]:
        raise ValueError("private labels not paired with public prompt IDs")
    for ref in private:
        if (hashlib.sha256(ref["chosen"].encode()).hexdigest()!=ref["reference_chosen_sha256"]
            or hashlib.sha256(ref["rejected"].encode()).hexdigest()!=ref["reference_rejected_sha256"]):
            raise ValueError("reference answer data modified since selection")
    return pub,private,generated


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--public-prompts",type=Path,required=True)
    p.add_argument("--private-pairs",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--generation-summary",type=Path,required=True)
    p.add_argument("--generation-checkpoint",type=Path,required=True)
    p.add_argument("--sft",type=Path,required=True)
    p.add_argument("--bf16-dpo",type=Path,required=True)
    p.add_argument("--fp32-dpo",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--smoke-pairs",type=int)
    a=p.parse_args()
    public,ref,gen=load_frozen(a.public_prompts,a.private_pairs,a.manifest,a.generation_summary,a.generation_checkpoint)
    if a.smoke_pairs is not None and not 1<=a.smoke_pairs<=8:
        raise ValueError("smoke only first1..8 frozen pairs")
    pairs=public[:a.smoke_pairs] if a.smoke_pairs else public
    for arm,path in (("sft",a.sft),("bf16_dpo",a.bf16_dpo),("fp32_dpo",a.fp32_dpo)):
        if hash_file(path/"adapter_model.safetensors")!=SHA[arm]:
            raise ValueError("model weights not frozen for evaluation")
    identity={
        "experiment":"EXP-014-HH-INDEPENDENT-HELDOUT",
        "dataset":"Anthropic/hh-rlhf helpful-base/test",
        "public_prompts_sha256":PUBLIC256_SHA,
        "private_pairs_sha256":PRIVATE256_SHA,
        "preference_labels_read_only_after_192_blind_generations":True,
        "generation_checkpoint_sha256":gen["generation_checkpoint_sha256"],
        "source_code_sha256":hash_file(Path(__file__)),
        "model_revision":MODEL_REV,
        "adapter_sha256":SHA,"max_total_tokens":MAX_TOTAL_TOKENS,
        "prompt_budget_tokens":448,"seed":42,
        "attempted_pairs":len(pairs),
        "subset":"fixed_first_n_smoke" if a.smoke_pairs else "full256",
        "not_temporally_new":True,
        "not_a_human_quality_rating_of_generated_responses":True,
    }
    a.output_dir.mkdir(parents=True,exist_ok=True)
    marker=a.output_dir/"identity.json"
    if marker.exists():
        if json.loads(marker.read_text())!=identity:raise ValueError("evaluation identity drift")
    else:marker.write_text(json.dumps(identity,indent=2)+"\n")
    checkpoint=a.output_dir/"checkpoint.jsonl"
    done={}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if not line.strip():continue
            row=json.loads(line);idx=int(row["index"])
            if idx in done or idx not in range(len(pairs)) or row["task_id"]!=pairs[idx]["task_id"]:
                raise ValueError("mutated/duplicate scoring checkpoint")
            done[idx]=row
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
    from peft import PeftModel
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_REV,local_files_only=True,use_fast=True)
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    base=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=MODEL_REV,quantization_config=q,device_map={"":0},
        dtype=torch.bfloat16,local_files_only=True)
    model=PeftModel.from_pretrained(base,str(a.sft),adapter_name="sft",is_trainable=False)
    model.load_adapter(str(a.bf16_dpo),adapter_name="bf16_dpo",is_trainable=False)
    model.load_adapter(str(a.fp32_dpo),adapter_name="fp32_dpo",is_trainable=False)
    model.eval();model.config.use_cache=False
    adapter_dtype={
        arm:sorted({str(w.dtype) for name,w in model.named_parameters()
                    if ".lora_" in name and f".{arm}." in name})
        for arm in ARMS
    }
    if any(adapter_dtype[arm]!=["torch.float32"] for arm in ARMS):
        raise ValueError("inference adapter storage dtype inconsistent")
    start=time.monotonic()
    with torch.inference_mode():
        for i,item in enumerate(pairs):
            if i in done:continue
            answer=ref[i]
            prefix=tokenizer.apply_chat_template(
                item["messages"],tokenize=False,add_generation_prompt=True)
            chosen=get_tokens(tokenizer,prefix,answer["chosen"],model.device)
            rejected=get_tokens(tokenizer,prefix,answer["rejected"],model.device)
            min_tokens=min(len(chosen[2]),len(rejected[2]))
            entry={
                "index":i,"task_id":item["task_id"],
                "scorable":bool(min_tokens),
                "chosen_target_tokens":len(chosen[2]),
                "rejected_target_tokens":len(rejected[2]),
                "chosen_truncated":bool(chosen[3]),
                "rejected_truncated":bool(rejected[3]),
            }
            if min_tokens:
                for arm in ARMS:
                    model.set_adapter(arm)
                    for choice,tokens in (("chosen",chosen),("rejected",rejected)):
                        ids,mask,positions,_=tokens
                        logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
                        picked=logits[:,[x-1 for x in positions],:].float()
                        labels=ids[:,positions]
                        nll=F.cross_entropy(
                            picked.transpose(1,2),labels,reduction="sum")
                        sumlogp=-float(nll.item())
                        entry[f"{arm}_{choice}_sum_logp"]=sumlogp
                        entry[f"{arm}_{choice}_mean_logp"]=sumlogp/len(positions)
                        del logits,picked,labels,nll
            else:
                entry["reason"]="one_candidate_had_zero_response_tokens"
            with checkpoint.open("a",encoding="utf-8") as h:
                h.write(json.dumps(entry,ensure_ascii=False)+"\n")
                h.flush()
            done[i]=entry
            if (i+1)%16==0 or i+1==len(pairs):
                print(f"EXP014 cross-source preference {i+1}/{len(pairs)}",flush=True)
    if set(done)!=set(range(len(pairs))):
        raise ValueError("not all frozen preference pairs scored")
    result={
        **identity,"status":"completed",
        "inference_adapter_dtypes":adapter_dtype,
        "metrics":summarize([done[i] for i in range(len(pairs))]),
        "checkpoint_sha256":hash_file(checkpoint),
        "GPU_peak_allocated_mib":round(torch.cuda.max_memory_allocated()/2**20,1),
        "execution_seconds":round(time.monotonic()-start,2),
        "limitations":"Public historical human preference labels, other model family distribution and variable answer lengths. No independent quality rating for the new model generations.",
    }
    (a.output_dir/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":main()
