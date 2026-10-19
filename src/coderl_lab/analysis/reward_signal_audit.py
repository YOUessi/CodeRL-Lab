"""Pure audit of complete EXP028v3 reward logs; never execute candidate code."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, pvariance, stdev

CLASSES=('all_correct','all_wrong','same_intermediate','nonzero_variance')
EXPECTED_MIXED={'narrow_seed43':28,'narrow_seed44':30,'broad_seed43':19,'broad_seed44':19}
COMPONENTS=('executable_fraction','dependency_fraction','semantic_fraction')
REQUIRED={'attempt_id','group_id','task_id','variant','reward','corrected_reward',
          *COMPONENTS,'code_sha256','tests_sha256'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def unique_object(pairs):
    out={}
    for key,value in pairs:
        if key in out:raise ValueError('duplicate JSON key: '+key)
        out[key]=value
    return out

def read_jsonl(path):
    result=[]
    def invalid(value):raise ValueError('nonfinite JSON: '+value)
    for i,line in enumerate(Path(path).read_text(encoding='utf8').splitlines(),1):
        if not line.strip():raise ValueError(f'blank JSONL record at line {i}')
        try:row=json.loads(line,parse_constant=invalid,object_pairs_hook=unique_object)
        except (ValueError,TypeError) as exc:raise ValueError(f'invalid JSONL line {i}: {exc}') from exc
        if not isinstance(row,dict):raise ValueError('JSONL record must be object')
        result.append(row)
    if not result:raise ValueError('empty JSONL')
    return result


def task_families(rows):
    mapping={}
    for row in rows:
        task=row.get('task_id');family=row.get('task_kind')
        if not isinstance(task,str) or not task or not isinstance(family,str) or not family:
            raise ValueError('missing task_id/task_kind')
        if task in mapping:raise ValueError('duplicate training task')
        mapping[task]=family
    if len(mapping)!=48:raise ValueError('expected48 frozen training tasks')
    return mapping


def fraction(value):
    if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:
        raise ValueError('invalid reward/component fraction')
    return float(value)


def classify(values):
    if len(set(values))>1:return 'nonzero_variance'
    if values[0]==1:return 'all_correct'
    if values[0]==0:return 'all_wrong'
    return 'same_intermediate'


def audit(rows,mapping,expected_groups=32):
    if type(expected_groups) is not int or expected_groups<1:raise ValueError('invalid group budget')
    if len(rows)!=expected_groups*4:raise ValueError('incomplete candidate budget')
    grouped=defaultdict(list);attempts=set()
    for position,row in enumerate(rows):
        if REQUIRED-row.keys():raise ValueError('missing required fields: '+str(sorted(REQUIRED-row.keys())))
        a=row['attempt_id'];g=row['group_id']
        if type(a) is not int or type(g) is not int or a!=position or g!=a//4:
            raise ValueError('noncontiguous, duplicate or reordered attempt/group IDs')
        if a in attempts:raise ValueError('duplicate attempt')
        attempts.add(a)
        if row['variant']!='corrected':raise ValueError('requires corrected full reward')
        if row['task_id'] not in mapping:raise ValueError('unknown training task')
        for field in ('code_sha256','tests_sha256'):
            value=row[field]
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('invalid source/test hash')
        e,d,s=(fraction(row[k]) for k in COMPONENTS)
        if s>e or e>d:raise ValueError('inconsistent execution/dependency/semantic fractions')
        reward=fraction(row['reward']);corrected=fraction(row['corrected_reward'])
        if reward!=corrected or reward!=round(.1*e+.1*d+.8*s,6):
            raise ValueError('full corrected reward formula mismatch')
        grouped[g].append(row)
    if set(grouped)!=set(range(expected_groups)):raise ValueError('missing groups')
    groups=[]
    for g,rr in sorted(grouped.items()):
        if len(rr)!=4 or len({r['task_id'] for r in rr})!=1 or len({r['tests_sha256'] for r in rr})!=1:
            raise ValueError('mixed/incomplete task or tests within group')
        rewards=[float(r['reward']) for r in rr];semantic=[float(r['semantic_fraction']) for r in rr]
        correct=[int(s==1.) for s in semantic];task=rr[0]['task_id']
        record={'group_id':g,'task_id':task,'family':mapping[task],'classification':classify(rewards),
                'reward_mean':mean(rewards),'reward_population_variance':pvariance(rewards),
                'reward_sample_std':stdev(rewards),'final_reward_zero_variance':len(set(rewards))==1,
                'semantic_all_correct':all(s==1 for s in semantic),'semantic_all_wrong':all(s==0 for s in semantic),
                'semantic_fraction_zero_variance':len(set(semantic))==1,
                'binary_correctness_zero_variance':len(set(correct))==1,
                'fully_correct_candidates':sum(correct)}
        for i,r in enumerate(rr):
            record[f'reward_{i}']=r['reward']
            for key in COMPONENTS:record[f'{key}_{i}']=r[key]
        groups.append(record)
    def aggregate(rr):
        counts=Counter(r['classification'] for r in rr)
        return {'groups':len(rr),'candidates':len(rr)*4,**{c:counts[c] for c in CLASSES},
                'final_reward_zero_variance':sum(r['final_reward_zero_variance'] for r in rr),
                'semantic_all_wrong':sum(r['semantic_all_wrong'] for r in rr),
                'semantic_fraction_zero_variance':sum(r['semantic_fraction_zero_variance'] for r in rr),
                'binary_correctness_zero_variance':sum(r['binary_correctness_zero_variance'] for r in rr),
                'fully_correct_candidates':sum(r['fully_correct_candidates'] for r in rr),
                'mean_reward':mean(r['reward_mean'] for r in rr)}
    families=[{'family':family,**aggregate([r for r in groups if r['family']==family])}
              for family in sorted(set(mapping.values())) if any(r['family']==family for r in groups)]
    return {'summary':aggregate(groups),'groups':groups,'families':families,
            'unobserved_families':sorted(set(mapping.values())-{r['family'] for r in groups})}


def write_csv(path,rows):
    with Path(path).open('x',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name',choices=tuple(EXPECTED_MIXED),required=True)
    p.add_argument('--reward-log',type=Path,required=True)
    p.add_argument('--train-data',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    report=audit(read_jsonl(args.reward_log),task_families(read_jsonl(args.train_data)))
    if report['summary']['nonzero_variance']!=EXPECTED_MIXED[args.run_name]:
        raise ValueError('historical mixed-group count differs from recorded audit')
    report.update({'run_name':args.run_name,'reward_log_sha256':sha(args.reward_log),
                   'train_data_sha256':sha(args.train_data),'audit_source_sha256':sha(__file__),
                   'reward_definition':'round(0.1*execution+0.1*dependency+0.8*semantic,6)',
                   'all_wrong_definition':'all four FINAL rewards exactly zero; semantic-all-wrong is separate',
                   'group_size':4,'expected_groups':32,'candidate_code_or_tests_exported':False})
    if args.output_dir.exists():raise FileExistsError('refuse overwrite')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    with (args.output_dir/'audit.json').open('x',encoding='utf8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    write_csv(args.output_dir/'groups.csv',report['groups'])
    write_csv(args.output_dir/'families.csv',report['families'])
    print(json.dumps({'run_name':args.run_name,**report['summary']},ensure_ascii=False,allow_nan=False))

if __name__=='__main__':main()
