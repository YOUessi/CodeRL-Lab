"""CPU-only paired bootstrap analysis of frozen TRAIN-011 heldout NLL checkpoint.

All 256 samples, including length-unscorable rows, must be accounted for.
"""
import argparse
import json
import random
from pathlib import Path

def paired_ci(rows: list[dict], *, iterations: int=20000, seed: int=42) -> dict:
    usable=[r for r in rows if int(r["target_tokens"])>0]
    if not usable: raise ValueError("no scoreable rows")
    n=len(usable)
    rng=random.Random(seed)
    effects=[]
    for _ in range(iterations):
        idx=(rng.randrange(n) for _ in range(n))
        total=0
        delta=0.0
        for j in idx:
            r=usable[j]
            total+=int(r["target_tokens"])
            delta+=float(r["sft_sum_nll"])-float(r["base_sum_nll"])
        effects.append(delta/total)
    effects.sort()
    return {
        "metric":"SFT - Base token-weighted heldout NLL",
        "paired_task_bootstrap_95_low":effects[int(0.025*iterations)],
        "paired_task_bootstrap_95_high":effects[int(0.975*iterations)-1],
        "iterations":iterations,"seed":seed,
        "scored_examples":n,
        "unscorable_prompt_length":len(rows)-n,
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--evaluation-dir",type=Path,required=True)
    a=p.parse_args()
    summary=json.loads((a.evaluation_dir/"summary.json").read_text())
    if summary.get("formal_full_256") is not True or summary.get("evaluated_protocol_version")!="length_512_v2_all_input_accounted":
        raise ValueError("full frozen v2 evaluation required")
    rows=[json.loads(line) for line in (a.evaluation_dir/"checkpoint.jsonl").read_text().splitlines() if line.strip()]
    if len(rows)!=256 or set(x["index"] for x in rows)!=set(range(256)):
        raise ValueError("every heldout row must have a unique checkpoint")
    result={
        "experiment":"TRAIN-011-HELDOUT-PAIRED-CI",
        "evaluation_protocol":"full256_length512_v2",
        "input_identity":json.loads((a.evaluation_dir/"identity.json").read_text()),
        "underlying_metrics":summary["metrics"],
        "uncertainty":paired_ci(rows),
        "limitations":"Model likelihood on reference completions, not real-user quality or code-task correctness. Some prompts are longer than 512 and have zero scoreable target tokens.",
    }
    (a.evaluation_dir/"paired_ci.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__":
    main()
