"""Public-only 7B capacity baseline; never read private test cases.

The pre-registered smoke uses four deterministic question IDs sorted from frozen v4.
Not an efficacy endpoint. Full evaluation must be pre-registered separately.
"""
import argparse
import json
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
from coderl_lab.analysis.livecodebench_official_eval import load_public_tasks,OfficialLiveCodeBenchExecutor,evaluate_code
from coderl_lab.datasets.livecodebench import sha256_file

MODEL="Qwen/Qwen2.5-7B-Instruct"
REVISION="a09a35458c702b33eeacc393d103063234e8bc28"
PUBLIC_SHA="62239669ec50b2ebee8cad1a0561e072e293a76a9a130d0b214bdbcedf980853"

def make_prompt(row):
    # A native Instruct prompt, deliberately not numerically comparable to the Qwen3 Base prompt.
    return [{"role":"system","content":"Solve the Python programming problem. Return only executable Python code without Markdown."},
        {"role":"user","content":str(row["question_content"])+("\nStarter code:\n"+str(row["starter_code"]) if row.get("starter_code") else "")}]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--public-tasks",type=Path,required=True)
    p.add_argument("--lcb-repo",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--smoke-tasks",type=int,default=4)
    args=p.parse_args()
    if sha256_file(args.public_tasks)!=PUBLIC_SHA: raise ValueError("frozen v4 public SHA mismatch")
    rows=load_public_tasks(args.public_tasks)
    if len(rows)!=101: raise ValueError("expected 101 public-only tasks")
    if not 1<=args.smoke_tasks<=101: raise ValueError("smoke task count invalid")
    ids=sorted(rows)[:args.smoke_tasks]
    if (args.output_dir/"summary.json").exists(): raise FileExistsError("refusing overwrite")
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    cfg=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,device_map="auto",dtype=torch.float16,quantization_config=cfg)
    model.eval()
    checker=OfficialLiveCodeBenchExecutor(livecodebench_repo=args.lcb_repo,timeout_seconds=6,memory_limit="1g")
    result=[]
    with torch.inference_mode():
        for i,qid in enumerate(ids,1):
            row=rows[qid]
            prompt=tokenizer.apply_chat_template(make_prompt(row),tokenize=False,add_generation_prompt=True)
            x=tokenizer(prompt,return_tensors="pt").to(model.device)
            start=time.monotonic()
            y=model.generate(**x,max_new_tokens=512,do_sample=False,pad_token_id=tokenizer.eos_token_id)
            output=tokenizer.decode(y[0,x["input_ids"].shape[1]:],skip_special_tokens=True).strip()
            # Public judge only. Private tests are NOT accessed.
            judged=evaluate_code(executor=checker,task=row,code=output)
            result.append({"question_id":qid,"public_pass":bool(judged.passed),"tokens":int(y.shape[1]-x["input_ids"].shape[1]),"seconds":round(time.monotonic()-start,2),"completion_sha256":__import__("hashlib").sha256(output.encode()).hexdigest()})
            print(f"7B public smoke {i}/{len(ids)}; passed={judged.passed}",flush=True)
    summary={"experiment":"EXP-7B-PUBLIC-SMOKE","model":MODEL,"model_revision":REVISION,"tasks":len(ids),"source_public_sha256":PUBLIC_SHA,"public_pass":sum(x["public_pass"] for x in result),"private_tests_accessed":False,"protocol":"7B instruct template greedy max512; descriptive public-only smoke","rows":result}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k!="rows"},indent=2))
if __name__=="__main__": main()
