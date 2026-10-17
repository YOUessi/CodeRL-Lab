"""EXP028 coverage intervention. Only trusted reference algorithms run on host.
Generated candidate programs and serialized reference source run ONLY in Docker.
No model outcome is used to select training or external tasks.
"""
from __future__ import annotations
import copy,hashlib,inspect,itertools,json,os,random
from collections import Counter,defaultdict
from pathlib import Path
from coderl_lab.datasets import train017_new_algorithms as old
from coderl_lab.train.train014b_reward_sandbox import validate_tests

NEW_FAMILIES=('prefix_threshold','sorted_gap','subset_mod','pair_distance','repeated_block','window_minimum')
PARAMS=(1,2,3,5)
SOURCE_TRAIN_SHA='21a63fcef2e74d7e902480db9aabceb6b61d5f3bd4aff551e1959f7b8e38e4bb'
EXPERIMENT='EXP028-fixed-coverage'

def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]

def new_oracle(kind,k,x):
    if kind not in NEW_FAMILIES or type(k) is not int or k<1:raise ValueError('unknown family/parameter')
    if kind=='prefix_threshold':
        return sum(sum(x[:j])>=k for j in range(1,len(x)+1))
    if kind=='sorted_gap':
        y=sorted(x)
        return sum(b-a>k for a,b in zip(y,y[1:]))
    if kind=='subset_mod':
        return sum(sum(x[j] for j in range(len(x)) if mask>>j&1)%k==0 for mask in range(1,1<<len(x)))
    if kind=='pair_distance':
        return sum(abs(a-b)<=k for a,b in itertools.combinations(x,2))
    if kind=='repeated_block':
        c=Counter(x[j:j+k] for j in range(max(0,len(x)-k+1)))
        return sum(n>=2 for n in c.values())
    return sum(min(x[j:j+k]) for j in range(max(0,len(x)-k+1)))

def new_reference(kind,k):
    sources={
      'prefix_threshold':f'def solve(x):\n total=0;count=0\n for value in x:\n  total+=value\n  count+=int(total>={k})\n return count\n',
      'sorted_gap':f'def solve(x):\n y=sorted(x)\n return sum(y[j+1]>y[j]+{k} for j in range(len(y)-1))\n',
      'subset_mod':f'def solve(x):\n counts=[1]+[0]*({k}-1)\n for value in x:\n  before=counts[:]\n  for remainder,n in enumerate(before):counts[(remainder+value)%{k}]+=n\n return counts[0]-1\n',
      'pair_distance':f'def solve(x):\n y=sorted(x);left=0;answer=0\n for right in range(len(y)):\n  while y[right]-y[left]>{k}:left+=1\n  answer+=right-left\n return answer\n',
      'repeated_block':f'def solve(x):\n seen=set();duplicate=set()\n for j in range(len(x)-{k}+1):\n  block=x[j:j+{k}]\n  if block in seen:duplicate.add(block)\n  seen.add(block)\n return len(duplicate)\n',
      'window_minimum':f'from collections import deque\ndef solve(x):\n q=deque();answer=0\n for i,value in enumerate(x):\n  while q and q[0]<=i-{k}:q.popleft()\n  while q and x[q[-1]]>=value:q.pop()\n  q.append(i)\n  if i+1>={k}:answer+=x[q[0]]\n return answer\n',
    }
    if kind not in sources:raise ValueError(kind)
    return sources[kind]

def new_prompt(kind,k):
    descriptions={
      'prefix_threshold':f'For an integer list x, count nonempty prefixes whose sum is at least {k}. Elements may be negative. Each prefix length is counted separately. Empty list returns 0.',
      'sorted_gap':f'For an integer list x, sort it in nondecreasing order, keeping duplicates, and count adjacent gaps strictly greater than {k}. Lists with fewer than two elements return 0.',
      'subset_mod':f'For an integer list x, count nonempty subsets of positions whose selected values sum to a multiple of {k}. Equal values at different positions remain distinct choices. Negative values are allowed. Empty list returns 0.',
      'pair_distance':f'For an integer list x, count pairs of distinct positions i<j with abs(x[i]-x[j]) <= {k}. Negative values and duplicates are allowed. Empty or singleton lists return 0.',
      'repeated_block':f'For an ASCII string x, count distinct substrings of length {k} that occur at least twice. Occurrences can overlap; count each distinct substring once. Return 0 when len(x)<{k}.',
      'window_minimum':f'For an integer list x, return the sum of the minimum value in every contiguous window of length {k}. Negative elements are allowed. Return 0 if len(x)<{k}.',
    }
    return 'Write Python source code only; define a top-level function solve(x) and return an integer. Do not include explanations or Markdown fences. '+descriptions[kind]

def examples(kind,k):
    rng=random.Random(int(hashlib.sha256(f'EXP028-DATA-v1|{kind}|{k}'.encode()).hexdigest()[:16],16))
    initial=['','a','aa','aaaa','abababa','abcabc','abc','abba','aabbcc'] if kind=='repeated_block' else [[],[0],[1],[-1],[1,1],[0,0,0],[-3,-1,0,2],[1,2,3],[-2,2,-2,2],[5,1,5,1]]
    values=[];seen=set()
    while len(values)<32:
        if initial:x=initial.pop(0)
        elif kind=='repeated_block':x=''.join(rng.choice('abcd') for _ in range(rng.randrange(1,25)))
        else:x=[rng.randrange(-5,9) for _ in range(rng.randrange(1,13))]
        key=canonical(x)
        if key not in seen:seen.add(key);values.append(x)
    return [{'x':x,'expected':new_oracle(kind,k,x)} for x in values]

def make_new_rows():
    rows=[];refs={}
    for kind in NEW_FAMILIES:
        for k in PARAMS:
            q=f'train/coverage/{kind}/{k}'
            row={'task_id':q,'task_kind':kind,'prompt':[{'role':'user','content':new_prompt(kind,k)}],'test_cases':examples(kind,k)}
            validate_tests(row['test_cases']);rows.append(row);refs[q]=new_reference(kind,k)
    return rows,refs

def build_arms(narrow):
    if len(narrow)!=48 or len({r['task_id'] for r in narrow})!=48:raise ValueError('require48 unique narrow tasks')
    families=defaultdict(list)
    for r in narrow:
        if set(r)!={'task_id','task_kind','prompt','test_cases'}:raise ValueError('unexpected task schema')
        validate_tests(r['test_cases']);families[r['task_kind']].append(r)
    if len(families)!=6 or any(len(v)!=8 for v in families.values()):raise ValueError('require six8-task families')
    retained=[]
    for kind in sorted(families):
        rr=sorted(families[kind],key=lambda r:r['task_id'])
        retained.extend(copy.deepcopy(rr[i]) for i in (0,2,4,6))
    added,refs=make_new_rows()
    if set(families)&set(NEW_FAMILIES):raise ValueError('new family overlap')
    arms={'narrow':copy.deepcopy(narrow),'broad':retained+added}
    if len({r['task_id'] for r in arms['broad']})!=48:raise ValueError('broad duplicate')
    return arms,refs

def write_jsonl(p,rows):
    with p.open('x',encoding='utf8') as f:
        for r in rows:f.write(canonical(r)+'\n')

def freeze_training(old_root,out):
    old_root=Path(old_root);out=Path(out)
    if out.exists():raise FileExistsError(out)
    if sha(old_root/'train.jsonl')!=SOURCE_TRAIN_SHA:raise ValueError('historical training source changed')
    original=read(old_root/'train.jsonl');arms,refs=build_arms(original)
    stage=out.with_name(out.name+'.stage')
    if stage.exists():raise FileExistsError(stage)
    stage.mkdir(parents=True)
    for r in original:
        kind=r['task_kind'];k=int(r['task_id'].rsplit('/',1)[1])
        refs[r['task_id']]='import math\n'+inspect.getsource(old.oracle)+f'\ndef solve(x):\n return oracle({kind!r},{k},x)\n'
    unique={r['task_id']:r for rr in arms.values() for r in rr}
    for arm,rr in arms.items():
        p=stage/arm;p.mkdir();write_jsonl(p/'train.jsonl',rr)
        for filename in ('holdout_public.jsonl','holdout_oracle.jsonl'):
            (p/filename).write_bytes((old_root/filename).read_bytes())
        manifests={'experiment':EXPERIMENT,'model_revision':old.REVISION,'train_tasks':48,'heldout_tasks':24,
                   'train_holdout_parameter_overlap':0,'heldout_oracle_never_seen_by_training':True,
                   'sha256':{'train':sha(p/'train.jsonl'),'holdout_public':sha(p/'holdout_public.jsonl'),'holdout_oracle':sha(p/'holdout_oracle.jsonl')}}
        (p/'manifest.json').write_text(canonical(manifests)+'\n')
    public=[{'task_id':q,'messages':r['prompt']} for q,r in sorted(unique.items())]
    write_jsonl(stage/'gate_public.jsonl',public)
    write_jsonl(stage/'gate_train_records.jsonl',[unique[q] for q in sorted(unique)])
    write_jsonl(stage/'references.jsonl',[{'task_id':q,'code':refs[q]} for q in sorted(refs)])
    manifest={'experiment':EXPERIMENT,'original_train_sha256':SOURCE_TRAIN_SHA,'rows_per_arm':48,'shared_tasks':24,'unique_training_tasks':72,
              'families':{a:dict(Counter(r['task_kind'] for r in rr)) for a,rr in arms.items()},
              'files':{str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file()}}
    (stage/'manifest.json').write_text(canonical(manifest)+'\n');os.rename(stage,out)
    return manifest

def freeze_external(source,previous,out):
    from coderl_lab.datasets.exp019b_mbppplus_freeze import eligible,SOURCE_SHA
    import pyarrow.parquet as pq
    source=Path(source);out=Path(out)
    if out.exists():raise FileExistsError(out)
    if sha(source)!=SOURCE_SHA:raise ValueError('MBPP+ source version drift')
    prior=[r for p in previous for r in read(p)]
    excluded={r['task_id'] for r in prior}
    if len(prior)!=216 or len(excluded)!=216:raise ValueError('require exactly216 unique previously used tasks')
    records=[]
    for raw in pq.read_table(source).to_pylist():
        item,reason=eligible(raw)
        if item is not None and item['public']['task_id'] not in excluded:records.append((item,raw))
    records.sort(key=lambda x:x[0]['order_key'])
    if len(records)<96:raise ValueError('insufficient untouched eligible tasks')
    chosen=records[:96];public=[x['public'] for x,r in chosen]
    private=[{'task_id':x['public']['task_id'],'entry_point':x['public']['entry_point'],'test':r['test'],'canonical_code':r['code']} for x,r in chosen]
    stage=out.with_name(out.name+'.stage')
    if stage.exists():raise FileExistsError(stage)
    stage.mkdir(parents=True);write_jsonl(stage/'public.jsonl',public);write_jsonl(stage/'private.jsonl',private)
    m={'experiment':'EXP028-unobserved-MBPP96','source_sha256':SOURCE_SHA,'excluded_task_count':216,'selected':96,
       'public_sha256':sha(stage/'public.jsonl'),'private_sha256':sha(stage/'private.jsonl'),
       'selection':'next96 in historical hash order, exclude all previous216; no verdict-based selection',
       'previous_public_sha256':[sha(p) for p in previous],'new_benchmark':False}
    (stage/'manifest.json').write_text(canonical(m)+'\n');os.rename(stage,out)
    return m
