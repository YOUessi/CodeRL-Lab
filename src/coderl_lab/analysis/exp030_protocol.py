"""EXP030 immutable evidence and public generation contract (standard library)."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

MODEL='Qwen/Qwen2.5-7B-Instruct'
REVISION='a09a35458c702b33eeacc393d103063234e8bc28'
SFT_SHA='25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8'
SEEDS=(51,52,53)
FINAL_MODELS={'sft'}|{f'seed{s}_step{k}' for s in SEEDS for k in (32,128)}
DEV_MODELS={'sft'}|{f'seed{s}_step{k}' for s in SEEDS for k in (32,64,128)}


def canonical(x):
    return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def rows(path):
    text=Path(path).read_text(encoding='utf8')
    if text and not text.endswith('\n'):raise ValueError('truncated JSONL; preserve evidence')
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def once(path,value):
    path=Path(path);body=canonical(value)+'\n'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_text(encoding='utf8')!=body:raise ValueError('immutable evidence changed: '+str(path))
    else:
        with path.open('x',encoding='utf8') as f:f.write(body)


def generation_protocol():
    return {'experiment':'EXP030','version':1,'model':MODEL,'revision':REVISION,
            'draws':4,'max_new_tokens':256,'temperature':0.8,'top_p':0.95,
            'sampling':'per_task_seed_num_return_sequences_4','seed_namespace':'EXP030|20261019|task_id',
            'inference_precision':'FP32-LoRA/NF4-BF16-base','max_prompt_tokens':4096,
            'normalizer':'DS1000-official-postprocess-at-b39aab71',
            'private_tests_in_generator':False,'all_final_generations_before_candidate_scoring':True}


def task_seed(task_id):
    return int(hashlib.sha256(('EXP030|20261019|'+task_id).encode()).hexdigest()[:8],16)%(2**31)


def public_rows(path):
    rr=rows(path);seen=set()
    for r in rr:
        if set(r)-{'task_id','source_task_id','messages','description','entry_point','signature'}:
            raise ValueError('private or unknown field in public input')
        q=r['task_id']
        if not isinstance(q,str) or q in seen:raise ValueError('duplicate or invalid public task')
        seen.add(q)
        if not isinstance(r['messages'],list) or not r['messages']:raise ValueError('missing prompt')
        for m in r['messages']:
            if set(m)!={'role','content'} or m['role'] not in {'system','user'} or not isinstance(m['content'],str):
                raise ValueError('invalid public message')
    if not rr:raise ValueError('empty task set')
    return rr


def validate_generations(rr,data,require_complete=False):
    expected={(r['task_id'],i) for r in data for i in range(4)};done={}
    for r in rr:
        k=(r['task_id'],r['sample_id'])
        if type(k[1]) is not int or k not in expected or k in done:raise ValueError('generation key mismatch')
        if r['seed']!=task_seed(k[0]) or text_sha(r['raw_code'])!=r['code_sha256']:
            raise ValueError('generation seed/code drift')
        if type(r['generated_tokens']) is not int or not 0<=r['generated_tokens']<=256:
            raise ValueError('invalid token count')
        if type(r['input_tokens']) is not int or not 0<r['input_tokens']<=4096:
            raise ValueError('invalid prompt length')
        if type(r['reached_token_limit']) is not bool:raise ValueError('invalid truncation flag')
        done[k]=r
    # A four-draw group is committed atomically; a partial task is never resumable.
    counts={q:sum(k[0]==q for k in done) for q,_ in expected}
    if any(n not in (0,4) for n in counts.values()):raise ValueError('partial generation group')
    if require_complete and set(done)!=expected:raise ValueError('incomplete generation')
    return done


def postprocess(code):
    """Exact string operations of the pinned official DS-1000 postprocessor."""
    if not isinstance(code,str):raise ValueError('solution must be text')
    return code.split('</code>')[0].replace('```python','').split('```')[0].split('\nEND SOLUTION')[0].replace('<code>','')


def snippet_payload(raw,oracle):
    """Construct inert source; the snippet and checker run only in the Docker driver."""
    code=postprocess(raw)
    wrapper='def __exp030_solution__():\n    return '+repr(code)+'\n'
    context=oracle['code_context']
    test=context+'\n__exp030_code = __exp030_solution__()\ntest_execution(__exp030_code)\n'
    if 'test_string(' in context:test+='test_string(__exp030_code)\n'
    return wrapper,{'entry_point':'__exp030_solution__','test':test}
