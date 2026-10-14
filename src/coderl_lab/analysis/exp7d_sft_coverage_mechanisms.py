"""EXP-7D: retrospective, read-only SFT coverage and diversity mechanism audit.

Source: fully frozen EXP-7C paired generation and Docker-scored checkpoints.
No model evaluation, dataset retrieval, sample selection, token reweighting or
changes to correct/incorrect labels occur here.

This is exploratory diagnosis of an already inspected historical benchmark;
statistics must not be called a new independent validation.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

ARMS = ("base", "sft")
EXPECTED_TASKS = 164
EXPECTED_SAMPLES = EXPECTED_TASKS * 4 * len(ARMS)
FROZEN_GENERATION_SHA = "91306aa04f61325692666a7e4ccd11c7fc462a2f7ec71751aafad5a29776a8c6"
FROZEN_SCORING_SHA = "a8b4c34985f35f5a442d067f80428b5105797f420043ddd8d5a26f0ace1814e3"


def file_sha(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def candidate_index(rows: list[dict], *, generation: bool) -> dict:
    indexed={}
    for record in rows:
        key=(str(record["task_id"]),str(record["arm"]),int(record["sample_id"]))
        if key in indexed:
            raise ValueError(f"duplicate {'generation' if generation else 'score'} key {key}")
        if key[1] not in ARMS or key[2] not in range(4):
            raise ValueError(f"unexpected arm/sample {key}")
        indexed[key]=record
    return indexed


def normalized_ast_key(row: dict) -> str | None:
    code=str(row["code"])
    try:
        tree=ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    return hashlib.sha256(
        ast.dump(tree,include_attributes=False).encode("utf-8")
    ).hexdigest()


def task_rows(generated: dict, scored: dict) -> list[dict]:
    if set(generated)!=set(scored) or len(generated)!=EXPECTED_SAMPLES:
        raise ValueError("full paired checkpoint keys must match")
    ids={key[0] for key in generated}
    if len(ids)!=EXPECTED_TASKS:
        raise ValueError("164 tasks required")
    expected={(qid,arm,n) for qid in ids for arm in ARMS for n in range(4)}
    if set(generated)!=expected:
        raise ValueError("each task must have exactly four samples in both arms")
    output=[]
    for qid in sorted(ids,key=lambda s:int(s.split("/")[1])):
        row={"task_id":qid,"arms":{}}
        for arm in ARMS:
            g=[generated[qid,arm,i] for i in range(4)]
            s=[scored[qid,arm,i] for i in range(4)]
            for ge,sc in zip(g,s):
                if ge["code_sha256"]!=sc["code_sha256"]:
                    raise ValueError("score does not correspond to generated code")
            result=[bool(x["passed"]) for x in s]
            ast_keys=[normalized_ast_key(x) for x in g]
            row["arms"][arm]={
                "correct":result,
                "solved_at_k":[any(result[:i]) for i in range(1,5)],
                "correct_samples":sum(result),
                "distinct_raw_codes":len({x["code_sha256"] for x in g}),
                "distinct_valid_ast":len({key for key in ast_keys if key is not None}),
                "syntax_valid_count":sum(key is not None for key in ast_keys),
                "output_tokens_mean":sum(int(x["output_tokens"]) for x in g)/4,
                "cap_hits":sum(int(x["output_tokens"])>=384 for x in g),
                "function_present":sum(bool(x["entry_point_defined"]) for x in g),
            }
        output.append(row)
    return output


def paired_ci(values: list[float], *, n_boot: int=20000, seed: int=42) -> dict:
    if len(values)!=EXPECTED_TASKS:
        raise ValueError("bootstrap must group all 164 tasks")
    rng=random.Random(seed)
    n=len(values)
    means=sorted(
        sum(values[rng.randrange(n)] for _ in range(n))/n
        for _ in range(n_boot)
    )
    return {
        "mean_paired_delta":sum(values)/n,
        "ci95_low":means[int(0.025*n_boot)],
        "ci95_high":means[int(0.975*n_boot)-1],
        "bootstrap_replicates":n_boot,
        "unit":"task", "seed":seed,
    }


def exact_mcnemar_two_sided(win: int, loss: int) -> float:
    """Exact binomial two-sided for discordant paired task-level outcomes."""
    from math import comb
    n=win+loss
    if n==0:
        return 1.
    tail=sum(comb(n,k) for k in range(min(win,loss)+1))/2**n
    return min(1.,2*tail)


def analyze(rows: list[dict]) -> dict:
    if len(rows)!=EXPECTED_TASKS:
        raise ValueError("incomplete per-task records")
    arm_stats={}
    for arm in ARMS:
        attr=[r["arms"][arm] for r in rows]
        arm_stats[arm]={
            "candidates":len(attr)*4,
            "sample_correct":sum(x["correct_samples"] for x in attr),
            "correct_candidates_by_sample_index":[sum(x["correct"][i] for x in attr) for i in range(4)],
            "solved_at_k":[sum(x["solved_at_k"][i] for x in attr) for i in range(4)],
            "mean_distinct_raw_code_per_task":sum(x["distinct_raw_codes"] for x in attr)/len(attr),
            "mean_distinct_python_ast_per_task":sum(x["distinct_valid_ast"] for x in attr)/len(attr),
            "mean_output_tokens":sum(x["output_tokens_mean"] for x in attr)/len(attr),
            "syntax_valid_candidates":sum(x["syntax_valid_count"] for x in attr),
            "top_level_entrypoint_present":sum(x["function_present"] for x in attr),
            "token_cap_hits":sum(x["cap_hits"] for x in attr),
            "correct_count_histogram":dict(sorted(Counter(x["correct_samples"] for x in attr).items())),
        }
    flips=Counter()
    transitions=Counter()
    for r in rows:
        base=r["arms"]["base"]
        sft=r["arms"]["sft"]
        flips[f"base_{int(base['solved_at_k'][3])}_sft_{int(sft['solved_at_k'][3])}"]+=1
        for i in range(4):
            transitions[f"base_{int(base['correct'][i])}_sft_{int(sft['correct'][i])}"]+=1
    lost=flips["base_1_sft_0"]
    gained=flips["base_0_sft_1"]
    pairs={
        "candidate_correct_fraction":[],
        "solved_at4_fraction":[],
        "distinct_raw_code_count_per_task":[],
        "distinct_valid_ast_count_per_task":[],
        "output_tokens_per_candidate":[],
        "syntax_valid_fraction":[],
    }
    for r in rows:
        base,sft=r["arms"]["base"],r["arms"]["sft"]
        pairs["candidate_correct_fraction"].append((sft["correct_samples"]-base["correct_samples"])/4)
        pairs["solved_at4_fraction"].append(float(sft["solved_at_k"][3])-float(base["solved_at_k"][3]))
        pairs["distinct_raw_code_count_per_task"].append(sft["distinct_raw_codes"]-base["distinct_raw_codes"])
        pairs["distinct_valid_ast_count_per_task"].append(sft["distinct_valid_ast"]-base["distinct_valid_ast"])
        pairs["output_tokens_per_candidate"].append(sft["output_tokens_mean"]-base["output_tokens_mean"])
        pairs["syntax_valid_fraction"].append((sft["syntax_valid_count"]-base["syntax_valid_count"])/4)
    return {
        "experiment":"EXP-7D-SFT-COVERAGE-MECHANISM-AUDIT",
        "status":"completed",
        "dataset":"HumanEval+ historical full164, previous EXP-7C",
        "qualification":"RETROSPECTIVE_READ_ONLY; neither new holdout nor causal proof of diversity-quality mechanism",
        "arms":arm_stats,
        "candidate_level_pair_transitions":dict(sorted(transitions.items())),
        "task_level_solved4_pair_transitions":dict(sorted(flips.items())),
        "exact_mcnemar_task_solved4_p":exact_mcnemar_two_sided(gained,lost),
        "paired_task_bootstrap":{
            name:paired_ci(values) for name,values in pairs.items()
        },
        "source_private_test_data_exported":False,
        "source_candidate_code_exported":False,
        "interpretive_boundary":"SFT increased code/AST variation in one 7B pair while decreasing correct sample fraction. Association is not a verified causal mediation relationship.",
    }


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--generation",type=Path,required=True)
    p.add_argument("--evaluation",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()

    gen_path=args.generation/"checkpoint.jsonl"
    scored_path=args.evaluation/"checkpoint.jsonl"
    if file_sha(gen_path)!=FROZEN_GENERATION_SHA:
        raise ValueError("EXP-7C generation checkpoint SHA mismatch")
    if file_sha(scored_path)!=FROZEN_SCORING_SHA:
        raise ValueError("EXP-7C scored checkpoint SHA mismatch")
    generation_summary=json.loads((args.generation/"summary.json").read_text())
    evaluation_summary=json.loads((args.evaluation/"summary.json").read_text())
    if generation_summary.get("sample_count")!=1312 or generation_summary.get("status")!="completed":
        raise ValueError("not full frozen 1312 generation samples")
    if evaluation_summary.get("samples")!=1312 or evaluation_summary.get("status")!="completed":
        raise ValueError("not full frozen 1312 scored samples")
    if evaluation_summary["frozen_inputs"]["generation_checkpoint_sha256"]!=FROZEN_GENERATION_SHA:
        raise ValueError("scorer used different generated code checkpoint")
    generated=candidate_index(read_jsonl(gen_path),generation=True)
    scored=candidate_index(read_jsonl(scored_path),generation=False)
    summary=analyze(task_rows(generated,scored))
    summary["input_sha256"]={
        "generation_checkpoint":FROZEN_GENERATION_SHA,
        "scoring_checkpoint":FROZEN_SCORING_SHA,
        "audit_source":file_sha(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():
        if json.loads(args.output.read_text())!=summary:
            raise FileExistsError("refusing overwrite of different audit result")
    else:
        args.output.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
