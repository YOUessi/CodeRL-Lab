"""EXP-014 frozen Anthropic HH-RLHF helpful-base public/label split.

Only public prompts (roles/messages) are accessible to model generation.
Chosen/rejected reference answers are stored in a separate local file and
are NEVER imported by the public prompt generator. Selection is based on
well-formed dialogue, old data overlap, tokenizer prompt budget, and stable
hash order, never on any model output or benchmark correctness.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

DATA_REVISION="65649da1780c54952392cb81400f0e9684641333"
DATA_SHA256="8be3fc1a13b27901631696f2be6f184c799f1baeaf145f53ac5db24960adc37b"
MODEL="Qwen/Qwen2.5-7B-Instruct"
MODEL_REV="a09a35458c702b33eeacc393d103063234e8bc28"
MARKER="\n\nAssistant:"
SPLIT=re.compile(r"\n\n(Human|Assistant):")
MAX_PROMPT_TOKENS=448
NUM_PAIRS=256
NUM_GENERATION=64


def hash_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda:stream.read(1024*1024),b""):
            h.update(data)
    return h.hexdigest()


def text_hash(text:str)->str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_text(s:str)->str:
    return re.sub(r"\s+"," ",s).casefold().strip()


def parse_pair(raw:dict)->dict|None:
    if set(raw)!={"chosen","rejected"}:
        return None
    chosen,rejected=raw["chosen"],raw["rejected"]
    if not isinstance(chosen,str) or not isinstance(rejected,str):
        return None
    a=chosen.rfind(MARKER)
    b=rejected.rfind(MARKER)
    if a<0 or b<0 or chosen[:a]!=rejected[:b]:
        return None
    chosen_answer=chosen[a+len(MARKER):].strip()
    rejected_answer=rejected[b+len(MARKER):].strip()
    if not chosen_answer or not rejected_answer or chosen_answer==rejected_answer:
        return None
    prompt=chosen[:a]
    matches=list(SPLIT.finditer(prompt))
    if not matches:
        return None
    roles=[m.group(1) for m in matches]
    if any(role!=("Human" if i%2==0 else "Assistant") for i,role in enumerate(roles)):
        return None
    if roles[-1]!="Human":
        return None
    contents=[
        prompt[m.end():matches[i+1].start()] if i<len(matches)-1 else prompt[m.end():]
        for i,m in enumerate(matches)
    ]
    if not all(v.strip() for v in contents):
        return None
    messages=[
        {"role":"user" if role=="Human" else "assistant","content":part.strip()}
        for role,part in zip(roles,contents)
    ]
    key=text_hash("EXP-014|seed=42|"+prompt)
    return {
        "task_id":"HH/"+key[:20],
        "sort_key":key,
        "context_sha256":text_hash(prompt),
        "messages":messages,
        "chosen":chosen_answer,
        "rejected":rejected_answer,
    }


def prior_last_user(prompt:str)->str:
    text=prompt.replace("\r\n","\n")
    if "### User:" in text:
        text=text.rsplit("### User:",1)[-1]
    if "### Assistant:" in text:
        text=text.split("### Assistant:")[0]
    return normalized_text(text)


def old_prompt_set(*paths:Path)->set[str]:
    prompts=set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(str(path))
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row=json.loads(line)
            v=row.get("prompt","")
            if not isinstance(v,str) or not v.strip():
                raise ValueError("old training preference missing prompt")
            prompts.add(prior_last_user(v))
    return prompts


def choose(source:list[dict], tokenizer, prior_prompts:set[str])->tuple[list[dict],dict]:
    stats=Counter()
    eligible=[]
    seen=set()
    for raw in source:
        stats["source_rows"]+=1
        parsed=parse_pair(raw)
        if parsed is None:
            stats["invalid_or_misaligned_dialogue"]+=1
            continue
        if parsed["context_sha256"] in seen:
            stats["repeated_context"]+=1
            continue
        seen.add(parsed["context_sha256"])
        last_user=normalized_text(parsed["messages"][-1]["content"])
        if last_user in prior_prompts:
            stats["overlap_with_previous_train_or_val"]+=1
            continue
        prompt=tokenizer.apply_chat_template(
            parsed["messages"],tokenize=False,add_generation_prompt=True,
        )
        plen=len(tokenizer(prompt,add_special_tokens=False)["input_ids"])
        if plen>MAX_PROMPT_TOKENS:
            stats["over_448_prompt_tokens"]+=1
            continue
        if plen==0 or len(last_user)<10:
            stats["empty_or_too_short_last_user"]+=1
            continue
        parsed["prompt_tokens"]=plen
        eligible.append(parsed)
        stats["eligible_before_sha_selection"]+=1
    eligible.sort(key=lambda row:row["sort_key"])
    if len(eligible)<NUM_PAIRS:
        raise ValueError("not enough source records for frozen 256 test pairs")
    selected=eligible[:NUM_PAIRS]
    if len({x["task_id"] for x in selected})!=NUM_PAIRS:
        raise ValueError("truncated public task hash collision")
    return selected,dict(stats)


def write_frozen(root:Path, selected:list[dict],stats:dict,source_path:Path)->dict:
    root.mkdir(parents=True,exist_ok=True)
    paths={
        "public_prompts":root/"public_prompts.jsonl",
        "private_pairs":root/"private_pairs.jsonl",
        "generation_prompts":root/"generation_prompts.jsonl",
        "manifest":root/"manifest.json",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("EXP-014 frozen dataset files already exist")
    public=[{
        "task_id":x["task_id"],
        "messages":x["messages"],
        "prompt_tokens":x["prompt_tokens"],
        "context_sha256":x["context_sha256"],
    } for x in selected]
    private=[{
        "task_id":x["task_id"],"chosen":x["chosen"],"rejected":x["rejected"],
        "reference_chosen_sha256":text_hash(x["chosen"]),
        "reference_rejected_sha256":text_hash(x["rejected"]),
    } for x in selected]
    def dump(path:Path,rows:list[dict]):
        with path.open("x",encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
    dump(paths["public_prompts"],public)
    dump(paths["private_pairs"],private)
    dump(paths["generation_prompts"],public[:NUM_GENERATION])
    data={
        "experiment":"EXP-014",
        "dataset":"Anthropic/hh-rlhf helpful-base/test.jsonl.gz",
        "dataset_revision":DATA_REVISION,
        "data_sha256":hash_file(source_path),
        "selection":"sha256(EXP-014|seed=42|full_common_context), sorted ascending",
        "data_validity_stats":stats,
        "frozen_preference_pairs":NUM_PAIRS,
        "frozen_generation_prompts":NUM_GENERATION,
        "max_prompt_tokens":MAX_PROMPT_TOKENS,
        "tokenizer_model":MODEL,"tokenizer_revision":MODEL_REV,
        "source_preference_labels_accessible_to_generation":False,
        "new_to_project":True,
        "independent_of_ultrafeedback_training_source":True,
        "public_historical_not_forward_temporal":True,
        "source_data_sha256":DATA_SHA256,
        "filenames":{
            key:path.name for key,path in paths.items() if key!="manifest"
        },
        "sha256":{
            key:hash_file(path) for key,path in paths.items() if key!="manifest"
        },
        "raw_text_exported_to_github":False,
    }
    paths["manifest"].write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
    return data


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-gz",type=Path,required=True)
    ap.add_argument("--ultrafeedback-train",type=Path,required=True)
    ap.add_argument("--ultrafeedback-validation",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    a=ap.parse_args()
    if hash_file(a.source_gz)!=DATA_SHA256:
        raise ValueError("pinned HH-RLHF original source hash mismatch")
    with gzip.open(a.source_gz,"rt",encoding="utf-8") as stream:
        source=[json.loads(line) for line in stream if line.strip()]
    if len(source)!=2354:
        raise ValueError(f"source row count drift: {len(source)}")
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(
        MODEL,revision=MODEL_REV,local_files_only=True,use_fast=True,
    )
    selected,stats=choose(
        source,tokenizer,
        old_prompt_set(a.ultrafeedback_train,a.ultrafeedback_validation),
    )
    result=write_frozen(a.output_dir,selected,stats,a.source_gz)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":main()
