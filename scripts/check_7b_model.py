"""Check whether the locally cached 7B policy can generate on Tang (no benchmark claims)."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    p.add_argument("--revision",default="b25037543e9394b818fdfca67ab2a00ecc7dd641")
    p.add_argument("--local-files-only",action="store_true",default=True)
    p.add_argument("--output",type=Path,default=Path("artifacts/exp7b/preflight.json"))
    args=p.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from coderl_lab.datasets.livecodebench import PINNED_LCB_DATASET_REVISION
    start=time.monotonic()
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=args.revision,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,revision=args.revision,local_files_only=True,device_map="auto",torch_dtype="auto")
    model.eval()
    messages=[{"role":"user","content":"Write a Python function add(a, b) that returns the sum of two integers. Reply with only the function code."}]
    prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    encoded=tokenizer(prompt,return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated=model.generate(**encoded,max_new_tokens=96,do_sample=False,pad_token_id=tokenizer.eos_token_id)
    output=tokenizer.decode(generated[0,encoded["input_ids"].shape[1]:],skip_special_tokens=True)
    result={"model":args.model,"revision":args.revision,"status":"loaded_and_generated","elapsed_seconds":round(time.monotonic()-start,2),"device":str(model.device),"gpu_peak_mib":round(torch.cuda.max_memory_allocated()/1024**2,1) if torch.cuda.is_available() else None,"generated_tokens":int(generated.shape[1]-encoded["input_ids"].shape[1]),"output":output,"note":"One trivial code-generation smoke; no pass@k or transfer conclusion.","lcb_source_revision":PINNED_LCB_DATASET_REVISION}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__":
    main()
