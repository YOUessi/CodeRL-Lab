"""Public-only inference for EXP028. This process never imports private oracles."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
from coderl_lab.analysis.exp028_protocol import (MODEL,REVISION,DRAWS,MAX_TOKENS,sha,rows,canonical,once,public_rows,sample_seed,validate_generations)


def generate(public_path,public_sha,adapter,adapter_sha,out):
    public_path=Path(public_path);adapter=Path(adapter);out=Path(out)
    if sha(public_path)!=public_sha or sha(adapter/'adapter_model.safetensors')!=adapter_sha:
        raise ValueError('model/public source hash mismatch')
    data=public_rows(public_path)
    out.mkdir(parents=True,exist_ok=True)
    identity={'experiment':'EXP028-public-generation','public_sha256':public_sha,'adapter_sha256':adapter_sha,
              'adapter_config_sha256':sha(adapter/'adapter_config.json'),'model':MODEL,'revision':REVISION,
              'source_sha256':sha(__file__),'draws':4,'max_new_tokens':256,'temperature':.8,'top_p':.95,
              'inference_precision':'FP32-LoRA/NF4-BF16-base','private_tests_opened':False,'task_count':len(data)}
    once(out/'identity.json',identity)
    cp=out/'checkpoint.jsonl';done,expected=validate_generations(rows(cp) if cp.exists() else [],data)
    if (out/'summary.json').exists():
        s=json.loads((out/'summary.json').read_text())
        if s['checkpoint_sha256']!=sha(cp) or set(done)!=expected:raise ValueError('completed generation drift')
        return s
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
    from peft import PeftModel
    if not torch.cuda.is_available():raise RuntimeError('real Tang CUDA required')
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,use_fast=True,local_files_only=True)
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    base=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,dtype=torch.bfloat16,device_map={'':0},quantization_config=quant)
    model=PeftModel.from_pretrained(base,str(adapter),is_trainable=False)
    model.eval();model.config.use_cache=True
    dtype=sorted({str(p.dtype) for n,p in model.named_parameters() if '.lora_' in n})
    if dtype!=['torch.float32']:raise ValueError('inference LoRA precision drift')
    started=time.monotonic();fresh=0
    with torch.inference_mode():
        for index,r in enumerate(data,1):
            prompt=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True)
            encoded=tok(prompt,return_tensors='pt').to(model.device)
            for i in range(DRAWS):
                key=r['task_id'],i
                if key in done:continue
                seed=sample_seed(*key);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                y=model.generate(**encoded,do_sample=True,temperature=.8,top_p=.95,max_new_tokens=MAX_TOKENS,use_cache=True,pad_token_id=tok.eos_token_id)
                answer_tokens=y[0,encoded['input_ids'].shape[1]:]
                text=tok.decode(answer_tokens,skip_special_tokens=True).strip()
                record={'task_id':key[0],'sample_id':i,'seed':seed,'raw_code':text,'code_sha256':hashlib.sha256(text.encode()).hexdigest(),
                        'generated_tokens':len(answer_tokens),'reached_token_limit':len(answer_tokens)>=MAX_TOKENS,
                        'input_tokens':int(encoded['input_ids'].shape[1])}
                with cp.open('a',encoding='utf8') as f:f.write(canonical(record)+'\n');f.flush()
                done[key]=record;fresh+=1
            print('GENERATED',index,'/',len(data),'CANDIDATES',len(done),'/',len(expected),flush=True)
    if set(done)!=expected:raise RuntimeError('incomplete generation')
    result={**identity,'status':'complete','candidate_count':len(done),'fresh_this_invocation':fresh,
            'checkpoint_sha256':sha(cp),'generated_tokens':sum(x['generated_tokens'] for x in done.values()),
            'reached_token_limit_count':sum(x['reached_token_limit'] for x in done.values()),
            'runtime_seconds':round(time.monotonic()-started,3),'adapter_dtype':dtype}
    once(out/'summary.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--public',type=Path,required=True);p.add_argument('--public-sha',required=True)
    p.add_argument('--adapter',type=Path,required=True);p.add_argument('--adapter-sha',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(canonical(generate(a.public,a.public_sha,a.adapter,a.adapter_sha,a.out)),flush=True)
if __name__=='__main__':main()
