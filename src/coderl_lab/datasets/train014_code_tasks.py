"""TRAIN-014 deterministic arithmetic-code tasks with train/heldout separation.

Generates 36 train prompts and 12 heldout prompts. Only train examples carry
test inputs/outputs to the GRPO reward hook; holdout test cases stay in a
separate frozen file and are NEVER added to GRPOTrainer's dataset.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
SEED = 42
INPUTS = (-17, -9, -5, -2, -1, 0, 1, 2, 5, 9, 14, 21)


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""):
            h.update(block)
    return h.hexdigest()


def eval_fn(kind: str, n: int, x: int) -> int:
    if kind == "add":
        return x+n
    if kind == "mul":
        return x*n
    if kind == "square_add":
        return x*x+n
    if kind == "abs_delta":
        return abs(x-n)
    if kind == "mod":
        return x % n
    if kind == "floor_div":
        return x // n
    raise ValueError("unknown task function; no arbitrary code execution")


def question(kind: str, n: int) -> str:
    meaning={
        "add": f"Return x plus {n}.",
        "mul": f"Return x multiplied by {n}.",
        "square_add": f"Return the square of x plus {n}.",
        "abs_delta": f"Return the absolute difference between x and {n}.",
        "mod": f"Return the Python remainder x % {n}.",
        "floor_div": f"Return the Python floor division x // {n}.",
    }[kind]
    return (
        "Write only executable Python source code. Define exactly a top-level "
        "function called solve that takes one integer argument x and returns "
        "an integer. Do not print or explain the answer. "
        "The function must handle negative integers correctly. " + meaning
    )


TRAIN_SPECS = {
    "add": [-10,-7,-3,1,3,6,10],
    "mul": [-9,-5,-2,2,4,7,11],
    "square_add": [-6,-3,0,2,5,8],
    "abs_delta": [-8,-3,1,4,8,12],
    "mod": [2,3,4,6,8],
    "floor_div": [2,3,4,5,6],
}
HOLDOUT_SPECS={
    "add":[-12,13],
    "mul":[-6,9],
    "square_add":[-9,11],
    "abs_delta":[-11,6],
    "mod":[5,9],
    "floor_div":[7,9],
}


def rows_from_specs(specs:dict, split:str) -> list[dict]:
    rows=[]
    for kind,params in sorted(specs.items()):
        for n in params:
            qid=f"{split}/{kind}/{n}"
            values=[
                {"x":x, "expected":eval_fn(kind,n,x)}
                for x in INPUTS
            ]
            rows.append({
                "task_id":qid,
                "prompt":[{"role":"user","content":question(kind,n)}],
                "test_cases":values,
                "task_kind":kind,
            })
    return rows


def write_jsonl(path:Path,rows:list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen task source: {path}")
    with path.open("x",encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+"\n")


def freeze(root:Path)->dict:
    root.mkdir(parents=True,exist_ok=True)
    train=rows_from_specs(TRAIN_SPECS,"train")
    holdout=rows_from_specs(HOLDOUT_SPECS,"heldout")
    # Split model-visible heldout task descriptions from answer/test oracle.
    holdout_public=[{k:r[k] for k in ("task_id","prompt","task_kind")} for r in holdout]
    holdout_oracle=[{k:r[k] for k in ("task_id","test_cases")} for r in holdout]
    keys={r["task_id"] for r in train}
    other={r["task_id"] for r in holdout}
    if len(keys)!=36 or len(other)!=12 or keys & other:
        raise ValueError("train/heldout task ID split integrity failed")
    train_keys={(r["task_kind"],r["task_id"].rsplit("/",1)[-1]) for r in train}
    hold_keys={(r["task_kind"],r["task_id"].rsplit("/",1)[-1]) for r in holdout}
    if train_keys & hold_keys:
        raise ValueError("heldout spec parameters overlap with training task")
    assert all(set(r)=={"task_id","prompt","task_kind"} for r in holdout_public)
    paths={
        "train":root/"train.jsonl",
        "holdout_public":root/"heldout_public.jsonl",
        "holdout_oracle":root/"heldout_oracle.jsonl",
    }
    for name,data in (
        ("train",train),("holdout_public",holdout_public),
        ("holdout_oracle",holdout_oracle)
    ):
        write_jsonl(paths[name],data)
    manifest={
        "experiment":"TRAIN-014-VERIFIABLE-GRPO",
        "source":"deterministic Python integer transformations",
        "source_script_sha256":digest(Path(__file__)),
        "model":MODEL,"model_revision":REVISION,
        "seed":SEED,
        "train_task_count":len(train),
        "heldout_task_count":len(holdout),
        "eval_inputs_per_task":len(INPUTS),
        "train_heldout_spec_overlap":0,
        "generation_prompt_does_not_contain_tests":True,
        "heldout_oracle_never_seen_by_GRPOTrainer":True,
        "task_family_count":len(TRAIN_SPECS),
        "sha256":{k:digest(v) for k,v in paths.items()},
        "warning":"These are deliberately small synthetic arithmetic-code functions, not high-difficulty real programming problems",
    }
    path=root/"manifest.json"
    if path.exists():raise FileExistsError("task source manifest already frozen")
    path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(freeze(a.output_dir),indent=2,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
