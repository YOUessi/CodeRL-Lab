"""EXP030 reference preflight and frozen-only candidate scoring in existing Docker jail."""
from __future__ import annotations
import argparse
import collections
import datetime
import json
import os
from pathlib import Path
import time
from coderl_lab.analysis.exp030_protocol import (
    canonical,once,rows,sha,text_sha,snippet_payload,public_rows,validate_generations,
    generation_protocol,FINAL_MODELS,DEV_MODELS,SEEDS,SFT_SHA,
)
from coderl_lab.datasets.exp030_ds1000 import validate_dataset,SOURCE_SHA256


def execution_protocol():
    from coderl_lab.analysis import exp029_frozen_eval as sandbox
    return {'experiment':'EXP030','task_adapter':'native_DS1000_test_execution_and_optional_test_string',
            'container_execution':sandbox.protocol(),'dataset_source_sha256':SOURCE_SHA256,
            'numpy_version':'2.2.6','pandas_version':'2.3.3','score_is_official_DS1000':False,
            'environment_revision':2,'numpy_select_compatibility':'coerce_choices_and_default_to_arrays_as_NumPy1.26',
            'normalization':generation_protocol()['normalizer'],'adapter_source_sha256':sha(__file__),
            'protocol_module_sha256':sha(Path(__file__).with_name('exp030_protocol.py'))}


def append(path,row):
    with Path(path).open('a',encoding='utf8') as f:
        f.write(canonical(row)+'\n');f.flush();os.fsync(f.fileno())


def brief(r):
    return {k:r[k] for k in ('passed','reason','resource_limited','key','record_sha256','cache_hit','container_executed')}


def reference_controls(dataset,out):
    """Allowed pre-training protocol validation; no model output is read or scored."""
    from coderl_lab.analysis import exp029_frozen_eval as sandbox
    from coderl_lab.analysis.exp028_reference_controls import control_summary
    dataset=Path(dataset);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    m=validate_dataset(dataset,private=True)
    identity={'experiment':'EXP030','stage':'reference_environment_preflight',
              'dataset_manifest_sha256':sha(dataset/'manifest.json'),'protocol':execution_protocol(),
              'model_candidates_read':0,'model_candidates_scored':0,'task_selection_by_reference_results':False}
    once(out/'identity.json',identity)
    hidden=rows(dataset/'private.jsonl');path=out/'reference_results.jsonl'
    old=rows(path) if path.exists() else [];done={r['task_id']:r for r in old}
    if len(done)!=len(old) or not set(done)<={r['task_id'] for r in hidden}:raise ValueError('reference checkpoint mismatch')
    for index,r in enumerate(hidden,1):
        code,oracle=snippet_payload(r['reference_code'],r)
        result=sandbox.score(code,oracle,out/'observations')
        record={'task_id':r['task_id'],'reference_code_sha256':text_sha(r['reference_code']),**brief(result)}
        if r['task_id'] in done:
            if done[r['task_id']]['record_sha256']!=record['record_sha256']:raise ValueError('reference evidence drift')
        else:append(path,record);done[r['task_id']]=record
        print('REFERENCE',index,'/',len(hidden),r['task_id'],result['reason'],flush=True)
    gate={**identity,**control_summary(list(done.values()),[r['task_id'] for r in hidden]),
          'identity_sha256':sha(out/'identity.json'),'records_sha256':sha(path),
          'per_task':list(done.values()),'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    # A complete reference run is immutable; do not refresh its timestamp on recheck.
    final=out/'final.json'
    if final.exists():
        prior=json.loads(final.read_text())
        if prior['identity_sha256']!=gate['identity_sha256'] or prior['records_sha256']!=gate['records_sha256']:
            raise ValueError('reference gate changed')
        return prior
    once(final,gate)
    return gate


def freeze_generations(dataset,model_manifest,out,split='final'):
    dataset=Path(dataset);model_manifest=Path(model_manifest);out=Path(out)
    data=public_rows(dataset/'public.jsonl');model_map=json.loads(model_manifest.read_text())
    expected=FINAL_MODELS if split=='final' else DEV_MODELS
    if set(model_map)!=expected:raise ValueError('incomplete model cohort')
    frozen={}
    for name,entry in sorted(model_map.items()):
        directory=Path(entry['generation_dir']);identity=json.loads((directory/'identity.json').read_text())
        summary=json.loads((directory/'summary.json').read_text());cp=directory/'checkpoint.jsonl'
        records=rows(cp);validate_generations(records,data,require_complete=True)
        if identity['protocol']!=generation_protocol() or identity['public_sha256']!=sha(dataset/'public.jsonl'):
            raise ValueError('generation protocol/public mismatch')
        if (summary['status']!='complete' or summary['checkpoint_sha256']!=sha(cp)
            or summary['identity_sha256']!=sha(directory/'identity.json')
            or summary['candidate_count']!=len(data)*4 or identity['adapter_sha256']!=entry['adapter_sha256']):
            raise ValueError('generation/model identity mismatch')
        if name=='sft' and identity['adapter_sha256']!=SFT_SHA:raise ValueError('SFT reference drift')
        frozen[name]={'adapter_sha256':identity['adapter_sha256'],'adapter_config_sha256':identity['adapter_config_sha256'],
                      'generation_dir':str(directory),'generation_sha256':sha(cp),
                      'identity_sha256':sha(directory/'identity.json'),'summary_sha256':sha(directory/'summary.json'),
                      'candidate_count':len(records)}
    freeze={'experiment':'EXP030','split':split,'status':'all_models_generated_frozen',
            'protocol':generation_protocol(),'public_sha256':sha(dataset/'public.jsonl'),
            'dataset_manifest_sha256':sha(dataset/'manifest.json'),'model_manifest_sha256':sha(model_manifest),
            'task_count':len(data),'models':frozen}
    once(out,freeze)
    return freeze


def score_frozen(dataset,freeze_path,reference_gate,out):
    from coderl_lab.analysis import exp029_frozen_eval as sandbox
    from coderl_lab.analysis import exp027_matched_eval as stats
    dataset=Path(dataset);freeze_path=Path(freeze_path);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    freeze=json.loads(freeze_path.read_text());data=public_rows(dataset/'public.jsonl')
    if (freeze['status']!='all_models_generated_frozen' or freeze['split']!='final'
        or set(freeze['models'])!=FINAL_MODELS or freeze['protocol']!=generation_protocol()
        or freeze['public_sha256']!=sha(dataset/'public.jsonl')
        or freeze['dataset_manifest_sha256']!=sha(dataset/'manifest.json')):
        raise ValueError('frozen final cohort/protocol changed')
    generated={}
    for name,e in freeze['models'].items():
        d=Path(e['generation_dir']);p=d/'checkpoint.jsonl'
        if sha(p)!=e['generation_sha256'] or sha(d/'identity.json')!=e['identity_sha256'] or sha(d/'summary.json')!=e['summary_sha256']:
            raise ValueError('generation evidence changed')
        generated[name]=rows(p);validate_generations(generated[name],data,require_complete=True)
    # No private data is read until every model's frozen output validates.
    m=validate_dataset(dataset,private=True)
    gate=json.loads(Path(reference_gate).read_text())
    if (gate['model_scoring_permitted'] is not True or gate['passed']!=m['task_count']
        or gate['dataset_manifest_sha256']!=sha(dataset/'manifest.json') or gate['protocol']!=execution_protocol()):
        raise ValueError('reference gate not valid for exact environment/data')
    oracle={r['task_id']:r for r in rows(dataset/'private.jsonl')}
    identity={'experiment':'EXP030','split':'final','generation_freeze_sha256':sha(freeze_path),
              'reference_gate_sha256':sha(reference_gate),'protocol':execution_protocol()}
    once(out/'identity.json',identity)
    judged={};started=time.monotonic()
    for name,rr in generated.items():
        sp=out/(name+'.jsonl');old=rows(sp) if sp.exists() else []
        done={(r['task_id'],r['sample_id']):r for r in old}
        expected={(r['task_id'],r['sample_id']) for r in rr}
        if len(done)!=len(old) or not set(done)<=expected:raise ValueError('scoring checkpoint mismatch')
        for index,g in enumerate(rr,1):
            key=(g['task_id'],g['sample_id']);code,test=snippet_payload(g['raw_code'],oracle[key[0]])
            verdict=sandbox.score(code,test,out/'observations')
            record={'task_id':key[0],'sample_id':key[1],'code_sha256':g['code_sha256'],**brief(verdict)}
            if key in done:
                if done[key]['record_sha256']!=record['record_sha256'] or done[key]['code_sha256']!=g['code_sha256']:
                    raise ValueError('scoring evidence drift')
            else:append(sp,record);done[key]=record
            if index%64==0:print('SCORED',name,index,'/',len(rr),flush=True)
        judged[name]=list(done.values())
    comparisons={str(s):stats.compare(stats.matrix(judged[f'seed{s}_step32']),stats.matrix(judged[f'seed{s}_step128'])) for s in SEEDS}
    result={**identity,'status':'complete','task_count':len(data),'models':freeze['models'],
            'metrics':{name:stats.metrics(rr) for name,rr in judged.items()},
            'primary_128_minus_32_by_seed':comparisons,
            'versus_sft':{name:stats.compare(stats.matrix(judged['sft']),stats.matrix(rr)) for name,rr in judged.items() if name!='sft'},
            'scoring_files_sha256':{name:sha(out/(name+'.jsonl')) for name in judged},
            'reference_controls':{'passed':gate['passed'],'total':gate['total'],'failed':gate['failed']},
            'candidate_positions_scored':sum(len(rr) for rr in judged.values()),
            'elapsed_this_invocation_seconds':round(time.monotonic()-started,3),
            'bootstrap_unit':'one independent DS1000 NumPy original task',
            'limits':['Custom87-task supported-library subset, not official DS1000',
                      'Three training seeds; per-seed task bootstrap does not estimate training-population uncertainty',
                      'No pretraining contamination guarantee; no checkpoint selection from final scores']}
    once(out/'final.json',result)
    return result


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('controls');c.add_argument('--dataset',type=Path,required=True);c.add_argument('--out',type=Path,required=True)
    f=sub.add_parser('freeze');f.add_argument('--dataset',type=Path,required=True);f.add_argument('--models',type=Path,required=True);f.add_argument('--out',type=Path,required=True);f.add_argument('--split',choices=['final','dev'],default='final')
    s=sub.add_parser('score');s.add_argument('--dataset',type=Path,required=True);s.add_argument('--freeze',type=Path,required=True);s.add_argument('--reference-gate',type=Path,required=True);s.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.command=='controls':r=reference_controls(a.dataset,a.out)
    elif a.command=='freeze':r=freeze_generations(a.dataset,a.models,a.out,a.split)
    else:r=score_frozen(a.dataset,a.freeze,a.reference_gate,a.out)
    print('EXP030_EVALUATION',canonical({k:r[k] for k in ('status','passed','failed','total','task_count','candidate_positions_scored') if k in r}),flush=True)


if __name__=='__main__':main()
