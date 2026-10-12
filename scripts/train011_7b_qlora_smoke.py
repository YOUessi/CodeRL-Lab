"""TRAIN-011 two-step real QLoRA gradient smoke on Tang.

Tiny hand-authored engineering fixtures ONLY. Not UltraChat training, not research efficacy.
Full training must use separately frozen dataset and validation policy.
"""
import argparse,json,math,hashlib
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path("artifacts/train011/gradient_smoke"))
    a=parser.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig,get_peft_model
    model_id="Qwen/Qwen2.5-7B-Instruct"
    revision="a09a35458c702b33eeacc393d103063234e8bc28"
    torch.manual_seed(42)
    cfg=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    tok=AutoTokenizer.from_pretrained(model_id,revision=revision,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(model_id,revision=revision,local_files_only=True,device_map="auto",dtype=torch.float16,quantization_config=cfg)
    model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.0,target_modules=["q_proj","v_proj"],task_type="CAUSAL_LM"))
    model.train()
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-5)
    fixtures=[("Explain what a Python list is.","A list is a mutable ordered collection."),
              ("What does len(x) return for a Python list?","It returns the number of elements in the list.")]
    metrics=[]
    for step,(question,answer) in enumerate(fixtures,1):
        prefix=tok.apply_chat_template([{"role":"user","content":question}],tokenize=False,add_generation_prompt=True)
        ids=tok(prefix+answer+tok.eos_token,return_tensors="pt",truncation=True,max_length=160,add_special_tokens=False)["input_ids"].to(model.device)
        prefix_len=len(tok(prefix,add_special_tokens=False)["input_ids"])
        labels=ids.clone()
        labels[:,:min(prefix_len,ids.shape[1]-1)]=-100
        optimizer.zero_grad(set_to_none=True)
        out=model(input_ids=ids,labels=labels)
        loss=out.loss
        if not torch.isfinite(loss): raise RuntimeError("nonfinite training loss")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.0)
        if not torch.isfinite(norm) or norm.item()<=0: raise RuntimeError("missing/nonfinite gradient")
        optimizer.step()
        row={"step":step,"loss":float(loss.item()),"grad_norm":float(norm.item())}
        metrics.append(row)
        print(row,flush=True)
    a.out.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(a.out/"adapter")
    weights=a.out/"adapter"/"adapter_model.safetensors"
    if not weights.exists(): raise RuntimeError("adapter checkpoint was not written")
    h=hashlib.sha256(weights.read_bytes()).hexdigest()
    result={"experiment":"TRAIN-011","status":"two_step_gradient_smoke","model":model_id,"revision":revision,
      "data":"2 hand-authored toy engineering examples; NOT UltraChat formal training",
      "optimizer_steps":2,"losses":metrics,"adapter_sha256":h,
      "cuda_peak_allocated_mib":round(torch.cuda.max_memory_allocated()/1024**2,2),
      "note":"Smoke validates finite loss, nonzero gradients, optimizer and adapter save only."}
    (a.out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__": main()
