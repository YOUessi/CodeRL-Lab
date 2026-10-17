"""EXP-024 freeze unseen MBPP+ cohort; separate public and private rows."""
import hashlib,json
from pathlib import Path
import pyarrow.parquet as pq
from coderl_lab.datasets.exp019b_mbppplus_freeze import eligible,digest,SOURCE_SHA
root=Path('artifacts/exp024')
out=root/'frozen_new96'
if out.exists():raise FileExistsError(out)
src=Path('artifacts/exp021/source/mbppplus_test.parquet')
if digest(src)!=SOURCE_SHA:raise ValueError('source changed')
old=[]
for p in (Path('artifacts/exp019b/frozen24/public24_prompts.jsonl'),Path('artifacts/exp021/frozen96/public96_prompts.jsonl')):
 old += [json.loads(x) for x in p.read_text().splitlines() if x]
assert len(old)==120
ids={x['task_id'] for x in old}
assert len(ids)==120
records=[]
for row in pq.read_table(src).to_pylist():
 item,reason=eligible(row)
 if item and item['public']['task_id'] not in ids:records.append((item,row))
records.sort(key=lambda x:x[0]['order_key'])
assert len(records)>=96
chosen=records[:96]
pub=[r[0]['public'] for r in chosen]
priv=[{'task_id':r[0]['public']['task_id'],'source_task_id':r[1]['task_id'],'entry_point':r[0]['public']['entry_point'],'test':r[1]['test'],'canonical_code':r[1]['code'],'source_sha256':hashlib.sha256(r[1]['code'].encode()).hexdigest()} for r in chosen]
assert not {x['task_id'] for x in pub}&ids
out.mkdir(parents=True)
for filename,rows in (('public96_prompts.jsonl',pub),('private96_tests.jsonl',priv)):
 with (out/filename).open('x') as f:
  for r in rows:f.write(json.dumps(r,sort_keys=True,ensure_ascii=False)+'\n')
m={'experiment':'EXP-024','source_sha256':SOURCE_SHA,'original_rows':378,'excluded_prior_tasks':120,'selected':96,'selection':'Next 96 in original EXP019B SHA order excluding both earlier cohorts','public_sha256':digest(out/'public96_prompts.jsonl'),'private_sha256':digest(out/'private96_tests.jsonl'),'source_test_inspected_for_selection':False}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print(json.dumps(m,indent=2))
