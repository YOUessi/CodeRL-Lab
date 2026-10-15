"""EXP-019 pinned MBPP+ independent-source 24-task benchmark freezing.

Uses ONLY task/format metadata (never model scores) for eligibility.
Public model prompts contain source problem and publicly advertised MBPP base
test examples to specify the function name. Hidden EvalPlus augmented test
and canonical reference implementation are in separate local files.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

EXPERIMENT="EXP-019-MBPPPLUS-THREE-7B"
DATA_REVISION="8a14ef6256e2d9ba80a03e9ce57ee349a69e6399"
SOURCE_SHA="dc20030b3788fccf617444edcb34138ef13d7e4fafd17bfcb8c1279dbb12399b"
DATA_FILE="data/test-00000-of-00001-d5781c9c51e02795.parquet"
SELECTED=24
ALL_SOURCE=378
SEED=20261015
MAX_PROMPT_CHARS=256


def digest(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda:f.read(1024*1024),b""):
            h.update(part)
    return h.hexdigest()


def stable_rank(task_id:int)->str:
    return hashlib.sha256(f"EXP019|seed={SEED}|{task_id}".encode()).hexdigest()


def select_public_data(rows:list[dict])->tuple[list[dict],list[dict],list[dict],dict]:
    if len(rows)!=ALL_SOURCE:
        raise ValueError("not official pinned MBPP+ 378 rows")
    if len({r["task_id"] for r in rows})!=ALL_SOURCE:
        raise ValueError("duplicate MBPP+ task ID")
    eligible=[]
    rejected={
        "nonempty_imports":0,"bad_original_problem_or_tests":0,"long_prompt":0,
    }
    for row in rows:
        prompt=row["prompt"]
        tests=row["test_list"]
        augmented=row["test"]
        code=row["code"]
        imports=row["test_imports"]
        if imports != []:
            rejected["nonempty_imports"]+=1
            continue
        if (not isinstance(prompt,str) or not prompt.strip()
            or not isinstance(tests,list) or len(tests)<3
            or not all(isinstance(x,str) and x.startswith("assert ") for x in tests)
            or not isinstance(augmented,str) or not augmented.strip()
            or not isinstance(code,str) or not code.strip()):
            rejected["bad_original_problem_or_tests"]+=1
            continue
        if len(prompt)>MAX_PROMPT_CHARS:
            rejected["long_prompt"]+=1
            continue
        eligible.append(row)
    eligible.sort(key=lambda row:stable_rank(row["task_id"]))
    if len(eligible)<SELECTED:
        raise ValueError("not enough stable benchmark tasks for fixed24")
    chosen=eligible[:SELECTED]
    public=[]
    oracle=[]
    references=[]
    for row in chosen:
        qid=f"MbppPlus/{row['task_id']}"
        text=(
            "Write executable Python source code only. Define all function(s) "
            "required by the problem and these PUBLIC examples, without any "
            "Markdown fences or explanations.\nProblem: "+row["prompt"].strip()+
            "\nPublic examples (these are provided as part of the specification):\n"+
            "\n".join(row["test_list"])
        )
        public.append({
            "task_id":qid,
            "prompt":[{"role":"user","content":text}],
            "source_task_id":int(row["task_id"]),
            "public_examples_sha256":hashlib.sha256(
                json.dumps(row["test_list"],ensure_ascii=False).encode()).hexdigest(),
        })
        oracle.append({
            "task_id":qid,
            "augmented_test":row["test"],
            "public_base_test_list":row["test_list"],
            "test_imports":row["test_imports"],
            "test_sha256":hashlib.sha256(row["test"].encode()).hexdigest(),
        })
        references.append({
            "task_id":qid,"reference_code":row["code"],
            "reference_sha256":hashlib.sha256(row["code"].encode()).hexdigest(),
        })
    if len({r["task_id"] for r in public})!=SELECTED:
        raise ValueError("selected MBPP task collision")
    if any(set(row)!={"task_id","prompt","source_task_id","public_examples_sha256"}
           for row in public):
        raise ValueError("private reference leaked into model prompt")
    stats={
        "original_rows":len(rows),
        "eligibility_counts":rejected,
        "eligible_before_SHA24":len(eligible),
        "chosen24_task_ids":[r["source_task_id"] for r in public],
        "selection_method":"sha256(EXP019|seed=20261015|task_id) ascending after format/import compatibility filter",
    }
    return public,oracle,references,stats


def freeze(source:Path,output_dir:Path)->dict:
    if digest(source)!=SOURCE_SHA:
        raise ValueError("pinned MBPP+ parquet sha256 mismatch")
    import pyarrow.parquet as pq
    rows=pq.read_table(source).to_pylist()
    public,oracle,reference,stats=select_public_data(rows)
    if output_dir.exists():
        raise FileExistsError("selected MBPP+ split already frozen; refusing overwrite")
    output_dir.mkdir(parents=True)
    files={
        "public_prompts":output_dir/"public_mbpp24.jsonl",
        "private_augmented_oracle":output_dir/"private_augmented_oracle.jsonl",
        "private_reference_code":output_dir/"private_reference_code.jsonl",
    }
    for name,data in (("public_prompts",public),
                       ("private_augmented_oracle",oracle),
                       ("private_reference_code",reference)):
        with files[name].open("x",encoding="utf-8") as out:
            for row in data:
                out.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
    summary={
        "experiment":EXPERIMENT,
        "dataset":"evalplus/mbppplus",
        "dataset_revision":DATA_REVISION,
        "source_file":DATA_FILE,
        "original_parquet_sha256":SOURCE_SHA,
        "script_source_sha256":digest(Path(__file__)),
        "source_rows":ALL_SOURCE,
        "selected_tasks":SELECTED,
        "sampling_seed":SEED,
        "selected_source_task_ids":stats["chosen24_task_ids"],
        "source_filter_stats":stats,
        "generation_prompts_have_base_test_examples":True,
        "private_augmented_oracles_not_in_generation_prompts":True,
        "private_reference_codes_not_in_generation_prompts":True,
        "only_metadata_eligibility_checks":True,
        "comparison_model_arms":["sft","previous_grpo","new_grpo"],
        "evaluation_seeds":[42,314159],
        "samples_per_task_per_seed_per_arm":2,
        "total_generations":SELECTED*2*2*3,
        "files":{name:path.name for name,path in files.items()},
        "sha256":{name:digest(path) for name,path in files.items()},
        "limitations":"MBPP+ is public historical benchmark, not a new time-split test. Deterministic selection does not exclude pretrained familiarity.",
    }
    (output_dir/"manifest.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(freeze(a.source,a.out),ensure_ascii=False,indent=2))
