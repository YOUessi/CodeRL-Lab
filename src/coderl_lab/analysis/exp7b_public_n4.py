"""Pre-registered 7B public-only multi-sampling evaluator with resumable output.

This evaluates PUBLIC testcase success, not private correctness or pass@k.
The historical v4 cohort is exploratory. Existing LCB v4/v5/v6 hidden outcomes
are never accessed or used to tune decoding and task selection.
"""
import argparse,hashlib,json,random
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
from coderl_lab.analysis.livecodebench_official_eval import load_public_tasks,OfficialLiveCodeBenchExecutor,evaluate_code
from coderl_lab.datasets.livecodebench import sha256_file
from coderl_lab.analysis.exp7b_public_smoke import MODEL,REVISION,PUBLIC_SHA,make_prompt

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--public-tasks",required=True,type=Path)
    p.add_argument("--lcb-repo",required=True,type=Path)
    p.add_argument("--output-dir",required=True,type=Path)
    p.add_argument("--samples-per-task",type=int,default=4)
    args=p.parse_args()
    if args.samples_per_task!=4: raise ValueError("frozen n=4")
    if sha256_file(args.public_tasks)!=PUBLIC_SHA: raise ValueError("public cohort changed")
    rows=load_public_tasks(args.public_tasks)
    if len(rows)!=101: raise ValueError("expected full historical v4 cohort")
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    checkpoint=out/"checkpoint.jsonl"
    done={}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            x=json.loads(line)
            key=(x["question_id"],x["sample_id"])
            if key in done: raise ValueError("duplicate checkpoint record")
            done[key]=x
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    cfg=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,device_map="auto",dtype=torch.float16,quantization_config=cfg)
    model.eval()
    judge=OfficialLiveCodeBenchExecutor(livecodebench_repo=args.lcb_repo,timeout_seconds=6,memory_limit="1g")
    for qid in sorted(rows):
        row=rows[qid]
        prompt=tok.apply_chat_template(make_prompt(row),tokenize=False,add_generation_prompt=True)
        encoded=tok(prompt,return_tensors="pt").to(model.device)
        for sample_id in range(4):
            key=(qid,sample_id)
            if key in done:continue
            torch.manual_seed(420000+int(hashlib.sha256(qid.encode()).hexdigest()[:8],16)%100000+sample_id)
            with torch.inference_mode():
                result=model.generate(**encoded,max_new_tokens=512,do_sample=True,temperature=0.8,top_p=0.95,pad_token_id=tok.eos_token_id)
            code=tok.decode(result[0,encoded["input_ids"].shape[1]:],skip_special_tokens=True).strip()
            outcome=evaluate_code(executor=judge,task=row,code=code)
            record={"question_id":qid,"sample_id":sample_id,"public_pass":bool(outcome.passed),"completion_sha256":hashlib.sha256(code.encode()).hexdigest(),"tokens":int(result.shape[1]-encoded["input_ids"].shape[1])}
            with checkpoint.open("a") as f:f.write(json.dumps(record)+"\n")
            done[key]=record
        print(f"public n4 completed task {qid} ({len(done)}/404)",flush=True)
    if len(done)!=404:raise RuntimeError("incomplete full cohort")
    grouped={}
    for (qid,_),r in done.items(): grouped.setdefault(qid,[]).append(r)
    summary={"experiment":"EXP-7B-V4-PUBLIC-N4","status":"exploratory_public_only","model":MODEL,"revision":REVISION,"cohort_sha":PUBLIC_SHA,"tasks":101,"samples":404,"public_sample_pass_fraction":sum(r["public_pass"] for r in done.values())/404,"public_solved_at4":sum(any(x["public_pass"] for x in group) for group in grouped.values()),"private_tests_accessed":False,"note":"Public tests only: not hidden Pass@4; historical v4, not prospectively held out."}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)
if __name__=="__main__":main()
