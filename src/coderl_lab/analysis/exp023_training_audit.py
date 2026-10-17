"""EXP-023 verify independent ablation training runs and reward variability.

Only reads three finalized training summaries, never benchmark oracles.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
ARMS=("format","execution","full")
EXPECTED_SOURCE="25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8"
EXPECTED_STEPS=32
EXPECTED_TRAIN_SHA="21a63fcef2e74d7e902480db9aabceb6b61d5f3bd4aff551e1959f7b8e38e4bb"
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()
def analyze(root:Path):
    results={}
    for arm in ARMS:
        path=root/f"grpo7b_{arm}_seed42"
        s=json.loads((path/"summary.json").read_text())
        ident=json.loads((path/"identity.json").read_text())
        attempts=[json.loads(x) for x in (path/"reward_attempts.jsonl").read_text().splitlines() if x.strip()]
        if (s.get("status")!="real_GPU_GRPO_completed"
            or s.get("reward_arm")!=arm
            or ident.get("reward_arm")!=arm
            or s["global_step"]!=EXPECTED_STEPS
            or s["train_sha256"]!=EXPECTED_TRAIN_SHA
            or s["warm_start_source_sft_adapter_sha256"]!=EXPECTED_SOURCE
            or s["saved_adapter_sha256"]!=sha(path/"adapter_model.safetensors")
            or s["reward_attempt_log_sha256"]!=sha(path/"reward_attempts.jsonl")
            or len(attempts)!=128):
            raise ValueError(f"unverified ablation arm: {arm}")
        if any(x["reward_arm"]!=arm for x in attempts):
            raise ValueError(f"mixed reward arms: {arm}")
        rewards=[float(x["reward"]) for x in attempts]
        groups={}
        for i in range(0,len(rewards),4):
            groups[i//4]=rewards[i:i+4]
        # Unlike task_id-global aggregation, this checks each contiguous GRPO group.
        nonconstant=sum(len(set(v))>1 for v in groups.values())
        results[arm]={
            "adapter_sha256":s["saved_adapter_sha256"],
            "changed_trainable_elements":s["initial_SFT_parameter_elements_changed_after_grpo"],
            "reward_histogram":s["reward_histogram"],
            "reward_mean_within_arm_only":s["mean_reward"],
            "nonconstant_contiguous_groups":nonconstant,
            "num_contiguous_groups":len(groups),
            "optimizer_steps":s["global_step"],
            "reward_calls":len(rewards),
            "gpu_peak_reserved_mib":s["gpu_peak_reserved_mib"],
        }
    return {"experiment":"EXP-023","status":"verified_training_only_not_generalization","arms":results,
            "warning":"Raw reward means are not comparable across different reward definitions. Contiguous groups should be confirmed against trainer actual batch/group ordering; no heldout benchmark scored here."}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=analyze(a.root)
    if a.out.exists():raise FileExistsError(a.out)
    a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
