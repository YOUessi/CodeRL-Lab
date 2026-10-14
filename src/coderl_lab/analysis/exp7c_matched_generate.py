"""EXP-7C matching 7B Instruct and 7B UltraChat-SFT HumanEval+ sampling.

No reference tests, canonical solutions or generated code are uploaded to GitHub.
All 164*4*2 raw candidate programs stay local until the generation freeze.
"""
from __future__ import annotations
import argparse
import ast
import contextlib
import hashlib
import json
import re
import time
from pathlib import Path

from coderl_lab.datasets.exp7c_humanevalplus import ALLOWED_FIELDS, sha256_file

MODEL = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
ADAPTER_SHA = "25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8"
SOURCE_SHA = "908377f1daf28dcb36846db73a5662b2e05a9907407c2696c89ad9d3b0b04492"
ARMS = ("base", "sft")
CONFIG = {
    "temperature": 0.8, "top_p": 0.95, "max_new_tokens": 384,
    "seed": 42, "precision": "nf4_double_quant_bf16",
    "prompt": "same_native_instruct_complete_python_function_v1",
    "extractor": "first_complete_leading_python_fence_else_preserve_v1",
}


def sha_text(v: str) -> str:
    return hashlib.sha256(v.encode("utf-8")).hexdigest()


def strip_outer_code_fence(raw: str) -> tuple[str, str]:
    text = raw.strip()
    ticks = re.escape(chr(96) * 3)
    match = re.match(r"^" + ticks + r"(?:python|py)?[ \t]*\r?\n", text, re.IGNORECASE)
    if not match:
        return text, "unfenced_unmodified"
    closing = re.search(r"\r?\n" + ticks + r"(?=[ \t]*(?:\r?\n|$))", text[match.end():])
    if closing:
        return text[match.end():match.end()+closing.start()].strip(), "first_closed_python_fence"
    return text, "unclosed_fence_preserved"


def structural_check(code: str, function_name: str) -> tuple[bool, bool]:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False, False
    found = any(
        isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))
        and x.name == function_name for x in tree.body
    )
    return True, found


def seed_for(task_id: str, sample_id: int) -> int:
    return int(sha_text(f"EXP-7C|seed=42|{task_id}|{sample_id}")[:8], 16) & 0x7fffffff


def prompt_messages(row: dict) -> list[dict]:
    if set(row) != ALLOWED_FIELDS:
        raise ValueError("forbidden oracle field entered generation prompt")
    return [
        {"role": "system", "content": (
            "Implement the required Python function as complete executable Python source. "
            "Include the exact function definition and necessary imports. "
            "Output ONLY Python code, with no Markdown fences or explanation."
        )},
        {"role": "user", "content": (
            "Complete the following function specification, returning the entire Python module "
            f"with a top-level function named {row['entry_point']}: \n\n" + row["prompt"]
        )},
    ]


def read_public(path: Path, manifest_path: Path) -> tuple[list[dict], dict]:
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    if any((
        m.get("experiment") != "EXP-7C",
        m.get("tasks") != 164,
        m.get("source_sha256") != SOURCE_SHA,
        m.get("source_revision") != "aa0d916268b1c17e84e881e9bd460508dd2fd308",
        m.get("public_sha256") != sha256_file(path),
        m.get("reference_solutions_exported") is not False,
        m.get("base_inputs_exported") is not False,
        m.get("plus_inputs_exported") is not False,
    )):
        raise ValueError("frozen prompt-only HumanEval+ input mismatch")
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if len(rows) != 164 or len({r["task_id"] for r in rows}) != 164:
        raise ValueError("must use 164 unique HumanEval IDs")
    if any(set(row) != ALLOWED_FIELDS for row in rows):
        raise ValueError("test/solution leaked into model inputs")
    return rows, m


def read_checkpoint(checkpoint: Path, rows: list[dict], n: int) -> dict:
    expected = {(r["task_id"], arm, sample) for r in rows for arm in ARMS for sample in range(n)}
    done = {}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            key = item["task_id"], item["arm"], int(item["sample_id"])
            if key in done or key not in expected:
                raise ValueError("checkpoint duplicate/unrecognized item")
            if (
                sha_text(item["raw"]) != item["raw_sha256"]
                or sha_text(item["code"]) != item["code_sha256"]
                or strip_outer_code_fence(item["raw"]) != (item["code"],item["extraction"])
                or item["seed"] != seed_for(item["task_id"],item["sample_id"])
            ):
                raise ValueError("checkpoint code/seed/normalizer changed")
            done[key] = item
    return done


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--public-tasks", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--adapter", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--smoke-tasks", type=int)
    p.add_argument("--samples-per-task", type=int, default=4)
    a=p.parse_args()
    if a.smoke_tasks is None and a.samples_per_task != 4:
        raise ValueError("formal experiment requires exactly 164 tasks x 4 samples")
    if a.smoke_tasks is not None and not (1 <= a.smoke_tasks <= 4 and a.samples_per_task == 1):
        raise ValueError("engineering smoke requires 1..4 tasks and 1 sample")
    all_rows, manifest = read_public(a.public_tasks,a.manifest)
    rows = all_rows if a.smoke_tasks is None else all_rows[:a.smoke_tasks]
    adapter_file = a.adapter / "adapter_model.safetensors"
    if not adapter_file.exists() or sha256_file(adapter_file) != ADAPTER_SHA:
        raise ValueError("TRAIN-011 7B SFT adapter SHA mismatch")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "experiment": "EXP-7C", "model": MODEL, "revision": REVISION,
        "adapter_sha256": ADAPTER_SHA,
        "public_sha256": manifest["public_sha256"],
        "generation_code_sha256": sha256_file(Path(__file__)),
        "settings": CONFIG, "samples_per_task": a.samples_per_task,
        "tasks": len(rows),
        "selection": "full164" if a.smoke_tasks is None else "smoke_first_n",
        "oracle_test_accessed_during_generation": False,
    }
    identity_path = a.output_dir / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("frozen checkpoint provenance does not match")
    else:
        identity_path.write_text(json.dumps(identity,indent=2)+"\n")
    checkpoint = a.output_dir / "checkpoint.jsonl"
    done = read_checkpoint(checkpoint,rows,a.samples_per_task)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16,
    )
    backbone=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=REVISION,local_files_only=True,
        quantization_config=quant,dtype=torch.bfloat16,device_map={"":0},
    )
    model=PeftModel.from_pretrained(backbone,str(a.adapter),is_trainable=False)
    model.eval()
    start=time.monotonic()
    total=len(rows)*len(ARMS)*a.samples_per_task
    for idx,row in enumerate(rows,1):
        prompt=tok.apply_chat_template(prompt_messages(row),tokenize=False,add_generation_prompt=True)
        encoded=tok(prompt,return_tensors="pt").to(model.device)
        for sample in range(a.samples_per_task):
            for arm in ARMS:
                key=row["task_id"],arm,sample
                if key in done:
                    continue
                seed=seed_for(row["task_id"],sample)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                disable=model.disable_adapter() if arm=="base" else contextlib.nullcontext()
                with torch.inference_mode(),disable:
                    generated=model.generate(
                        **encoded,max_new_tokens=CONFIG["max_new_tokens"],
                        temperature=CONFIG["temperature"],top_p=CONFIG["top_p"],
                        do_sample=True,pad_token_id=tok.eos_token_id,
                    )
                raw=tok.decode(generated[0,encoded["input_ids"].shape[1]:],skip_special_tokens=True)
                code,reason=strip_outer_code_fence(raw)
                syntax_ok,entry_ok=structural_check(code,row["entry_point"])
                record={
                    "task_id":row["task_id"],"sample_id":sample,"arm":arm,"seed":seed,
                    "entry_point":row["entry_point"],"raw":raw,"raw_sha256":sha_text(raw),
                    "code":code,"code_sha256":sha_text(code),"extraction":reason,
                    "python_ast_valid":syntax_ok,"entry_point_defined":entry_ok,
                    "output_tokens":int(generated.shape[1]-encoded["input_ids"].shape[1]),
                }
                with checkpoint.open("a",encoding="utf-8") as stream:
                    stream.write(json.dumps(record,ensure_ascii=False)+"\n")
                    stream.flush()
                done[key]=record
        if idx%5==0 or idx==len(rows):
            print(f"EXP-7C: {idx}/{len(rows)} tasks, {len(done)}/{total} samples frozen",flush=True)
    if len(done)!=total:
        raise RuntimeError("incomplete n4 matched generation")
    summary={
        **identity,"status":"completed","sample_count":len(done),
        "checkpoint_sha256":sha256_file(checkpoint),
        "by_arm":{arm:{
            "samples":sum(r["arm"]==arm for r in done.values()),
            "python_ast_valid":sum(r["arm"]==arm and r["python_ast_valid"] for r in done.values()),
            "entrypoint_present":sum(r["arm"]==arm and r["entry_point_defined"] for r in done.values()),
            "output_cap_hits":sum(r["arm"]==arm and r["output_tokens"]>=384 for r in done.values()),
        } for arm in ARMS},
        "gpu_peak_allocated_mib":round(torch.cuda.max_memory_allocated()/1024**2,1),
        "elapsed_seconds_current_invocation":round(time.monotonic()-start,2),
        "evaluation_oracle_accessed":False,
    }
    (a.output_dir/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
