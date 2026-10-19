"""Public-only EXP030 inference; four draws per task, no checker imports or reads."""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import time
from coderl_lab.analysis.exp030_protocol import (
    MODEL,REVISION,canonical,once,rows,sha,text_sha,public_rows,task_seed,
    generation_protocol,validate_generations,
)


def generate(public,public_sha,adapter,adapter_sha,out):
    public=Path(public);adapter=Path(adapter);out=Path(out)
    if sha(public)!=public_sha or sha(adapter/'adapter_model.safetensors')!=adapter_sha:
        raise ValueError('public/model hash mismatch')
    data=public_rows(public)
    identity={'experiment':'EXP030-public-generation','protocol':generation_protocol(),
              'public_sha256':public_sha,'adapter_sha256':adapter_sha,
              'adapter_config_sha256':sha(adapter/'adapter_config.json'),'source_sha256':sha(__file__),
              'protocol_module_sha256':sha(Path(__file__).with_name('exp030_protocol.py')),'task_count':len(data)}
    if out.exists():raise FileExistsError('generation output must be a fresh directory; preserve prior attempts')
    out.mkdir(parents=True);once(out/'identity.json',identity)
    started=time.monotonic();records=[]
    try:
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
        from peft import PeftModel
        if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
        torch.cuda.reset_peak_memory_stats()
        tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,use_fast=True,local_files_only=True)
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
        base=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,
              dtype=torch.bfloat16,device_map={'':0},quantization_config=quant)
        model=PeftModel.from_pretrained(base,str(adapter),is_trainable=False)
        model.eval();model.config.use_cache=True
        if sorted({str(p.dtype) for n,p in model.named_parameters() if '.lora_' in n})!=['torch.float32']:
            raise ValueError('inference LoRA precision drift')
        eos=model.generation_config.eos_token_id
        eos_ids={eos} if isinstance(eos,int) else set(eos or [tok.eos_token_id])
        load_seconds=time.monotonic()-started
        with torch.inference_mode():
            for index,r in enumerate(data,1):
                prompt=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True)
                encoded=tok(prompt,return_tensors='pt').to(model.device)
                input_tokens=int(encoded['input_ids'].shape[1])
                if input_tokens>4096:raise ValueError('prompt exceeds frozen token limit; no silent truncation')
                seed=task_seed(r['task_id']);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                group_start=time.monotonic()
                y=model.generate(**encoded,do_sample=True,temperature=.8,top_p=.95,
                                 max_new_tokens=256,num_return_sequences=4,use_cache=True,
                                 pad_token_id=tok.eos_token_id)
                if int(y.shape[0])!=4:raise RuntimeError('incomplete draw batch')
                group=[]
                for i in range(4):
                    tokens=y[i,input_tokens:].tolist();stop=next((j for j,t in enumerate(tokens) if t in eos_ids),None)
                    actual=tokens if stop is None else tokens[:stop+1]
                    raw=tok.decode(actual,skip_special_tokens=True).strip()
                    group.append({'task_id':r['task_id'],'sample_id':i,'seed':seed,'raw_code':raw,
                                  'code_sha256':text_sha(raw),'generated_tokens':len(actual),
                                  'input_tokens':input_tokens,'reached_token_limit':stop is None and len(actual)>=256,
                                  'termination':'eos' if stop is not None else 'length'})
                validate_generations(group,[r],require_complete=True)
                once(out/'groups'/f'{index:04d}.json',{'records':group,'elapsed_seconds':round(time.monotonic()-group_start,4)})
                records.extend(group)
                print('GENERATED',index,'/',len(data),'CANDIDATES',len(records),'/',len(data)*4,flush=True)
        validate_generations(records,data,require_complete=True)
        cp=out/'checkpoint.jsonl'
        with cp.open('x',encoding='utf8') as f:
            f.write(''.join(canonical(r)+'\n' for r in records));f.flush();os.fsync(f.fileno())
        result={'experiment':'EXP030','status':'complete','identity_sha256':sha(out/'identity.json'),
                'checkpoint_sha256':sha(cp),'candidate_count':len(records),'task_count':len(data),
                'elapsed_seconds':round(time.monotonic()-started,3),'model_load_seconds':round(load_seconds,3),
                'generated_tokens':sum(r['generated_tokens'] for r in records),
                'length_truncated_candidates':sum(r['reached_token_limit'] for r in records),
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
                'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        once(out/'summary.json',result);return result
    except BaseException as exc:
        once(out/'failure.json',{'experiment':'EXP030','error_type':type(exc).__name__,'reason':str(exc),
             'completed_candidates':len(records),'elapsed_seconds':round(time.monotonic()-started,3),
             'automatic_retry':False})
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--public',type=Path,required=True);p.add_argument('--public-sha',required=True)
    p.add_argument('--adapter',type=Path,required=True);p.add_argument('--adapter-sha',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(canonical(generate(a.public,a.public_sha,a.adapter,a.adapter_sha,a.out)),flush=True)


if __name__=='__main__':main()
