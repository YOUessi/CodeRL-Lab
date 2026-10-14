"""EXP-014B read-only 224-to-512 response-budget engineering comparison.

Validates all 192 model completions for both runs, including exact model
identity, input prompt hashes, per-task seeds and answer hashes. Reports
only truncation, length and identical response counts. No private HH labels,
no generated text, no quality or human ratings are exported.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import statistics
from pathlib import Path

from coderl_lab.datasets.exp014_hh_helpful import hash_file

ARMS=("sft","bf16_dpo","fp32_dpo")
SRC64="c58be29d6040c31e8a1340ca54f7056cf5ff9fc2bfb71ff901433b1114eae557"


def digest(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def freeze_check(folder:Path, expected_cap:int, expected_exp:str) -> tuple[dict,dict]:
    summary=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
    identity=json.loads((folder/"identity.json").read_text(encoding="utf-8"))
    checkpoint=folder/"generation_checkpoint.jsonl"
    if (summary.get("status")!="generated_blinded_needs_human_review"
        or summary.get("answers_generated")!=192
        or summary.get("blinded_packets")!=64
        or summary.get("human_review_status")!="not_reviewed"
        or summary.get("generation_checkpoint_sha256")!=hash_file(checkpoint)
        or summary.get("source_prompts_sha256")!=SRC64
        or summary.get("reference_labels_never_accessed") is not True
        or summary.get("num_prompts")!=64
        or summary.get("num_outputs")!=192
        or summary.get("config",{}).get("max_new_tokens")!=expected_cap
        or summary.get("experiment")!=expected_exp
        or identity.get("experiment")!=expected_exp
        or identity.get("config")!=summary.get("config")
        or summary.get("arms_sha256")!=identity.get("arms_sha256")
    ):
        raise ValueError(f"incomplete or changed {expected_exp} generation identity")
    packet=folder/"blind_review_packets.jsonl"
    key=folder/"blind_assignment_key.jsonl"
    if hash_file(packet)!=summary.get("review_packet_sha256"):
        raise ValueError("review packet SHA mismatch")
    if hash_file(key)!=summary.get("key_sha256"):
        raise ValueError("model assignment key SHA mismatch")
    packet_lines=[json.loads(x) for x in packet.read_text().splitlines() if x.strip()]
    key_lines=[json.loads(x) for x in key.read_text().splitlines() if x.strip()]
    if len(packet_lines)!=len(key_lines)!=64:
        raise ValueError("not 64 blinded packets")
    if len(packet_lines)!=64 or len(key_lines)!=64:
        raise ValueError("not 64 blinded packets")
    for p,k in zip(packet_lines,key_lines):
        if (p["task_id"]!=k["task_id"]
            or set(p["candidate_answers"])!={"A","B","C"}
            or "mapping" in p
            or set(k["mapping"].values())!=set(ARMS)):
            raise ValueError("reviewer packet identity may have leaked")
        if any(digest(p["candidate_answers"][letter])!=k["answer_sha256"][letter]
               for letter in ("A","B","C")):
            raise ValueError("reviewer packet answer changed")
    records=[json.loads(line) for line in checkpoint.read_text().splitlines() if line.strip()]
    by_key={}
    for item in records:
        k=item["task_id"],item["arm"]
        if (k in by_key or k[1] not in ARMS
            or digest(item["answer"])!=item["answer_sha256"]):
            raise ValueError("duplicate or modified generated answer")
        by_key[k]=item
    if len(by_key)!=192 or len({qid for qid,arm in by_key})!=64:
        raise ValueError("not all 64x3 outputs")
    return summary,by_key


def compare(old_dir:Path,new_dir:Path)->dict:
    a,old=freeze_check(old_dir,224,"EXP-014-HH-BLIND-GENERATION")
    b,new=freeze_check(new_dir,512,"EXP-014B-LONG-HH-BLIND-GENERATION")
    if set(old)!=set(new):
        raise ValueError("different tasks or models in length comparison")
    config1={k:v for k,v in a["config"].items() if k!="max_new_tokens"}
    config2={k:v for k,v in b["config"].items() if k!="max_new_tokens"}
    if (config1!=config2 or a["arms_sha256"]!=b["arms_sha256"]
        or a["data_public_sha256"]!=b["data_public_sha256"]
        or a["source_prompts_sha256"]!=b["source_prompts_sha256"]):
        raise ValueError("budget not the only model sampling difference")
    for k in old:
        if (old[k]["seed"]!=new[k]["seed"]
            or old[k]["context_sha256"]!=new[k]["context_sha256"]):
            raise ValueError("mismatched seeds/context for paired sample")
    per_arm={}
    for arm in ARMS:
        old_rows=[old[k] for k in sorted(old) if k[1]==arm]
        new_rows=[new[k] for k in sorted(new) if k[1]==arm]
        per_arm[arm]={
            "matched":len(old_rows),
            "old_224_cap_hits":sum(int(r["generated_tokens"]>=224) for r in old_rows),
            "new_512_cap_hits":sum(int(r["generated_tokens"]>=512) for r in new_rows),
            "old_mean_tokens":statistics.mean(r["generated_tokens"] for r in old_rows),
            "new_mean_tokens":statistics.mean(r["generated_tokens"] for r in new_rows),
            "new_nonempty_answers":sum(bool(r["answer"].strip()) for r in new_rows),
            "identical_unchanged_text":sum(x["answer_sha256"]==y["answer_sha256"]
                                           for x,y in zip(old_rows,new_rows)),
        }
    result={
        "experiment":"EXP-014B-LONG-REPLY-ENGINEERING-AUDIT",
        "status":"completed_frozen",
        "qualification":"posthoc engineering repair, not independent benchmark or human quality test",
        "task_count":64,
        "matched_model_answers":192,
        "same_prompt_seed_models_and_sampler_except_cap":True,
        "old_cap_tokens":224,"new_cap_tokens":512,
        "old_cap_hits_total":sum(v["old_224_cap_hits"] for v in per_arm.values()),
        "new_cap_hits_total":sum(v["new_512_cap_hits"] for v in per_arm.values()),
        "cap_hits_reduced":sum(v["old_224_cap_hits"]-v["new_512_cap_hits"] for v in per_arm.values()),
        "per_arm":per_arm,
        "old_checkpoint_sha256":a["generation_checkpoint_sha256"],
        "new_checkpoint_sha256":b["generation_checkpoint_sha256"],
        "new_blind_review_packet_sha256":b["review_packet_sha256"],
        "new_blind_assignment_key_sha256":b["key_sha256"],
        "quality_judgements":0,
        "human_review_status":"not_reviewed",
        "raw_responses_exported_to_github":False,
        "limitation":"Reaching the cap is a proxy for possible truncation, not certainty that an answer is incomplete. Lower cap-hit frequency does not demonstrate helpfulness.",
    }
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--old-dir",type=Path,required=True)
    ap.add_argument("--new-dir",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    result=compare(args.old_dir,args.new_dir)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.out.exists():
        if json.loads(args.out.read_text())!=result:
            raise FileExistsError("refusing overwrite of immutable evaluation report")
    else:
        args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
