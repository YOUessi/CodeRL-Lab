"""EXP-019B: freeze a 24-task MBPP+ cross-source coding benchmark.

Selection reads only task IDs, prompt text and canonical function signatures
for model-visible inputs. It never selects by execution success or uses
hidden tests to choose/construct the public prompt cohort.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

SOURCE_SHA="dc20030b3788fccf617444edcb34138ef13d7e4fafd17bfcb8c1279dbb12399b"
SOURCE_ROWS=378
COHORT=24
EXPERIMENT="EXP-019B-MBPPPLUS-EXTERNAL"
REVISION="Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28"


def digest(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1048576),b""):
            h.update(chunk)
    return h.hexdigest()


def eligible(row:dict)->tuple[dict|None,str]:
    qid=row["task_id"]
    prompt=row["prompt"]
    source=row["code"]
    if type(qid) is not int or not isinstance(prompt,str) or not prompt.strip() or len(prompt)>600:
        return None,"bad_id_or_prompt"
    try: tree=ast.parse(source)
    except (SyntaxError,ValueError):
        return None,"unparseable_source"
    funcs=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))]
    if len(funcs)!=1 or not isinstance(funcs[0],ast.FunctionDef):
        return None,"not_one_regular_top_level_function"
    fn=funcs[0]
    if fn.decorator_list or fn.name.startswith("_"):
        return None,"unsupported_signature"
    signature="def "+fn.name+"("+ast.unparse(fn.args)+")"
    if fn.returns:
        signature+=" -> "+ast.unparse(fn.returns)
    signature+=":"
    if len(signature)>220:
        return None,"very_long_signature"
    order=hashlib.sha256(f"EXP019B|seed=42|{qid}".encode()).hexdigest()
    visible={
        "task_id":f"MBPPPlus/{qid}",
        "source_task_id":qid,
        "entry_point":fn.name,
        "signature":signature,
        "description":prompt,
        "messages":[{
            "role":"user",
            "content":(
                "Write only valid executable Python source code, with no markdown "
                "and no explanation. "+prompt.strip()+"\n\n"
                "Use exactly the required function declaration (you may add type "
                "annotations only when compatible):\n"+signature+"\n"
                "Implement the body and any necessary imports. "
                "Do not rely on hidden tests."
            )
        }],
    }
    return {"order_key":order, "public":visible},"eligible"


def freeze(source:Path,out:Path)->dict:
    if digest(source)!=SOURCE_SHA:
        raise ValueError("MBPP+ Parquet source SHA mismatch")
    import pyarrow.parquet as pq
    tbl=pq.read_table(source)
    if tbl.num_rows!=SOURCE_ROWS:
        raise ValueError("MBPP+ original 378 row count changed")
    fields=set(tbl.column_names)
    if not {"task_id","prompt","code","test","test_list","test_imports"}.issubset(fields):
        raise ValueError("MBPP+ schema is incomplete")
    original=tbl.to_pylist()
    found=[]
    reasons=Counter()
    seen=set()
    for row in original:
        item,reason=eligible(row)
        reasons[reason]+=1
        if item is None:
            continue
        task_id=item["public"]["task_id"]
        if task_id in seen:
            raise ValueError("duplicate MBPP task ID")
        seen.add(task_id)
        found.append((item,row))
    found.sort(key=lambda pair:pair[0]["order_key"])
    if len(found)<COHORT:
        raise ValueError("not enough structurally eligible source tasks")
    cohort=found[:COHORT]
    public=[pair[0]["public"] for pair in cohort]
    private=[
        {
            "task_id":pair[0]["public"]["task_id"],
            "source_task_id":pair[1]["task_id"],
            "entry_point":pair[0]["public"]["entry_point"],
            "test":pair[1]["test"],
            "canonical_code":pair[1]["code"],
            "source_sha256":hashlib.sha256(pair[1]["code"].encode()).hexdigest(),
        } for pair in cohort
    ]
    root=out
    if root.exists():
        raise FileExistsError("EXP019B previously frozen source exists; not overwriting")
    root.mkdir(parents=True)
    paths={
        "public":root/"public24_prompts.jsonl",
        "private":root/"private24_tests.jsonl",
    }
    for name,rows in (("public",public),("private",private)):
        with paths[name].open("x",encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
    result={
        "experiment":EXPERIMENT,
        "original_dataset":"evalplus/mbppplus test parquet",
        "original_source_sha256":SOURCE_SHA,
        "original_total":SOURCE_ROWS,
        "selection":"First 24 after sorting eligible tasks by SHA256('EXP019B|seed=42|task_id')",
        "eligibility_reason_counts":dict(reasons),
        "eligible_total":len(found),
        "selected_task_count":len(public),
        "model_prompt_contains_only_original_description_and_canonical_function_signature":True,
        "canonical_function_body_and_test_assertions_not_in_public_prompts":True,
        "source_tests_never_used_for_selecting_tasks":True,
        "revision":REVISION,
        "test_environment":"Pinned semaremask-evalplus Docker with isolated 15-second fixed test-field script; not official leaderboard scoring",
        "public_prompt_sha256":digest(paths["public"]),
        "private_test_sha256":digest(paths["private"]),
        "source_script_sha256":digest(Path(__file__)),
        "historical_public_benchmark_not_forward_temporal":True,
        "test_full_reference_code_stays_only_on_Tang":True,
    }
    (root/"manifest.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(freeze(a.source,a.out),indent=2,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
