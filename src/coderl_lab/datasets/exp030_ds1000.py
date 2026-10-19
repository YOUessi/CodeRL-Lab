"""Freeze all supported independent NumPy originals, with no score-based selection."""
from __future__ import annotations
import argparse
import ast
import gzip
import json
import re
from pathlib import Path
from coderl_lab.analysis.exp030_protocol import canonical,once,rows,sha,text_sha,public_rows

SOURCE_REPO='xlang-ai/DS-1000'
SOURCE_REVISION='b39aab71da6d23ef8d3cac59a7c5f834516ab334'
SOURCE_SHA256='e8c6daa9d7223976bce0296644f3933f78d7f47830669ff05cd61da62c6ba9b3'
SUPPORTED_IMPORTS={'numpy','pandas','copy','tokenize','io'}
SYSTEM='Complete the Python solution requested by the user. Return only the code needed after BEGIN SOLUTION, with no explanation.'


def imports(r):
    code='\n'.join(r[k] for k in ('prompt','reference_code','code_context'))
    modules={a or b for a,b in re.findall(r'(?:^|\n)\s*(?:from\s+(\w+)(?:\.\w+)*\s+import\b|import\s+(\w+)\b)',code)}
    for node in ast.walk(ast.parse(r['code_context'])):
        if isinstance(node,ast.Import):modules.update(a.name.split('.')[0] for a in node.names)
        elif isinstance(node,ast.ImportFrom) and node.module:modules.add(node.module.split('.')[0])
    return modules


def eligible(r):
    m=r['metadata']
    return m['library']=='Numpy' and m['perturbation_type']=='Origin' and not imports(r)-SUPPORTED_IMPORTS


def write_rows(path,data):
    body=''.join(canonical(x)+'\n' for x in data)
    if path.exists():
        if path.read_text()!=body:raise ValueError('frozen dataset changed')
    else:
        with path.open('x',encoding='utf8') as f:f.write(body)


def freeze_source(source,out,source_sha,expected_count=87):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    selected=sorted((r for r in source if eligible(r)),key=lambda r:r['metadata']['problem_id'])
    ids=[r['metadata']['problem_id'] for r in selected]
    groups=[(r['metadata']['library'],r['metadata']['perturbation_origin_id']) for r in selected]
    if len(ids)!=expected_count or len(set(ids))!=len(ids) or len(set(groups))!=len(ids):
        raise ValueError('unexpected independent task cohort')
    public=[{'task_id':f'DS1000/{r["metadata"]["problem_id"]}',
             'source_task_id':r['metadata']['problem_id'],
             'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':r['prompt']}]} for r in selected]
    private=[{'task_id':f'DS1000/{r["metadata"]["problem_id"]}',
              'reference_code':r['reference_code'],'code_context':r['code_context'],
              'metadata':r['metadata']} for r in selected]
    write_rows(out/'public.jsonl',public);write_rows(out/'private.jsonl',private)
    excluded=[{'problem_id':r['metadata']['problem_id'],'unsupported_imports':sorted(imports(r)-SUPPORTED_IMPORTS)}
              for r in source if r['metadata']['library']=='Numpy' and r['metadata']['perturbation_type']=='Origin' and not eligible(r)]
    manifest={'experiment':'EXP030','source_repository':SOURCE_REPO,'source_revision':SOURCE_REVISION,
              'source_file':'data/ds1000.jsonl.gz','source_sha256':source_sha,
              'selection':'all_Numpy_Origin_with_supported_imports_before_reference_or_model_scores',
              'supported_imports':sorted(SUPPORTED_IMPORTS),'task_count':len(ids),'selected_ids':ids,
              'independent_origins':len(set(groups)),'excluded_originals':excluded,
              'public_sha256':sha(out/'public.jsonl'),'private_sha256':sha(out/'private.jsonl'),
              'reference_programs_not_executed_during_preparation':True,
              'model_pretraining_contamination_unknown':True}
    once(out/'manifest.json',manifest)
    return manifest


def validate_dataset(root,private=False):
    root=Path(root);m=json.loads((root/'manifest.json').read_text())
    if sha(root/'public.jsonl')!=m['public_sha256']:raise ValueError('public data hash mismatch')
    rr=public_rows(root/'public.jsonl')
    if len(rr)!=m['task_count'] or [r['source_task_id'] for r in rr]!=m['selected_ids']:
        raise ValueError('public cohort mismatch')
    if private:
        if sha(root/'private.jsonl')!=m['private_sha256']:raise ValueError('private data hash mismatch')
        hidden=rows(root/'private.jsonl')
        if [r['task_id'] for r in hidden]!=[r['task_id'] for r in rr]:raise ValueError('private cohort mismatch')
    return m


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if sha(a.source)!=SOURCE_SHA256:raise ValueError('pinned compressed source hash mismatch')
    with gzip.open(a.source,'rt',encoding='utf8') as f:source=[json.loads(x) for x in f]
    if len(source)!=1000:raise ValueError('source count drift')
    print(canonical(freeze_source(source,a.out,SOURCE_SHA256)))


if __name__=='__main__':main()
