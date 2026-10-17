"""EXP028 preregistered pure gates. No model, Docker or private-test imports."""
from __future__ import annotations
import hashlib,json,math
from collections import defaultdict
from pathlib import Path

MODEL='Qwen/Qwen2.5-7B-Instruct'
REVISION='a09a35458c702b33eeacc393d103063234e8bc28'
SFT_SHA='25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8'
ARMS=('narrow','broad')
SEEDS=(43,44)
DRAWS=4
STEPS=32
MAX_TOKENS=256

def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def sample_seed(task,sample):return int(hashlib.sha256(f'EXP028|{task}|{sample}|42'.encode()).hexdigest()[:8],16)%(2**31-1)

def once(path,value):
    path=Path(path);text=canonical(value)+'\n'
    if path.exists():
        if path.read_text()!=text:raise ValueError('immutable artifact changed: '+str(path))
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',encoding='utf8') as f:f.write(text)


def gate(arms):
    if set(arms)!=set(ARMS):raise ValueError('need both registered arms')
    result={'experiment':'EXP028','passed':True,'arms':{},'stop_reasons':[],
            'thresholds':{'min_mixed':8,'min_new_mixed':4,'min_syntax_fraction':.9,'max_clip_fraction':.2}}
    for arm,rr in arms.items():
        groups=defaultdict(list);keys=set()
        for r in rr:
            key=(r['task_id'],r['sample_id'])
            if key in keys or type(key[1]) is not int or key[1] not in range(4):raise ValueError('duplicate/bad sample')
            keys.add(key)
            if not math.isfinite(r['reward']) or not 0<=r['reward']<=1:raise ValueError('invalid reward')
            for field in ('syntax_valid','clipped','is_new'):
                if type(r[field]) is not bool:raise ValueError('invalid gate flag')
            groups[r['task_id']].append(r)
        if len(rr)!=192 or len(groups)!=48 or any(len(g)!=4 for g in groups.values()):raise ValueError('incomplete gate')
        mixed=sum(len({x['reward'] for x in g})>1 for g in groups.values())
        newmixed=sum(all(x['is_new'] for x in g) and len({x['reward'] for x in g})>1 for g in groups.values())
        syntax=sum(x['syntax_valid'] for x in rr)/192
        clipped=sum(x['clipped'] for x in rr)/192
        r={'tasks':48,'candidates':192,'mixed_groups':mixed,'new_mixed_groups':newmixed,
           'syntax_fraction':syntax,'clip_fraction':clipped}
        result['arms'][arm]=r
        for bad,why in ((mixed<8,'insufficient_group_signal'),(arm=='broad' and newmixed<4,'new_families_no_signal'),
                        (syntax<.9,'syntax_gate_failed'),(clipped>.2,'token_limit_gate_failed')):
            if bad:result['stop_reasons'].append(arm+':'+why)
    result['passed']=not result['stop_reasons']
    return result


def public_rows(path):
    data=rows(path);seen=set()
    for r in data:
        if set(r)-{'task_id','messages','description','entry_point','signature','source_task_id'}:
            raise ValueError('nonpublic field in generator input')
        if r['task_id'] in seen:raise ValueError('duplicate public task')
        seen.add(r['task_id'])
        if not isinstance(r['messages'],list) or not r['messages']:raise ValueError('missing messages')
        for x in r['messages']:
            if set(x)!={'role','content'} or not isinstance(x['content'],str):raise ValueError('invalid message')
    return data


def validate_generations(data,public):
    expected={(r['task_id'],i) for r in public for i in range(4)};found={}
    for r in data:
        key=(r['task_id'],r['sample_id'])
        if key not in expected or key in found or r['seed']!=sample_seed(*key):raise ValueError('generation key mismatch')
        if hashlib.sha256(r['raw_code'].encode()).hexdigest()!=r['code_sha256']:raise ValueError('generation hash mismatch')
        found[key]=r
    return found,expected
