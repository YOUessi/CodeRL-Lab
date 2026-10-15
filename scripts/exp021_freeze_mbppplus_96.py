#!/usr/bin/env python3
"""EXP-021 prepare next 96 MBPP+ tasks from the identical pinned 378-row Parquet.
Public prompts are written separately from private tests. Never select by verdict.
"""
import argparse, ast, hashlib, json
from pathlib import Path
import pyarrow.parquet as pq

SOURCE_SHA="dc20030b3788fccf617444edcb34138ef13d7e4fafd17bfcb8c1279dbb12399b"
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for c in iter(lambda:f.read(1<<20),b""):h.update(c)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser()
 a.add_argument("--source",required=True,type=Path)
 a.add_argument("--previous-public",required=True,type=Path)
 a.add_argument("--out",required=True,type=Path)
 args=a.parse_args()
 if sha(args.source)!=SOURCE_SHA:raise ValueError("MBPP+ dataset changed: fail closed")
 if args.out.exists():raise FileExistsError("Never overwrite a frozen expansion")
 prev=[json.loads(s) for s in args.previous_public.read_text().splitlines() if s]
 assert len(prev)==24 and len({x["task_id"] for x in prev})==24
 excluded={x["task_id"] for x in prev}
 rows=pq.read_table(args.source).to_pylist()
 assert len(rows)==378
 from coderl_lab.datasets.exp019b_mbppplus_freeze import eligible
 eligible_rows=[]
 for r in rows:
  item,reason=eligible(r)
  if item is not None and item["public"]["task_id"] not in excluded:
   eligible_rows.append((item,r))
 eligible_rows.sort(key=lambda x:x[0]["order_key"])
 assert len(eligible_rows)>=96
 chosen=eligible_rows[:96]
 public=[v["public"] for v,_ in chosen]
 private=[{"task_id":v["public"]["task_id"],"source_task_id":r["task_id"],"entry_point":v["public"]["entry_point"],"test":r["test"],"canonical_code":r["code"],"source_sha256":hashlib.sha256(r["code"].encode()).hexdigest()} for v,r in chosen]
 assert not ({x["task_id"] for x in public}&excluded)
 # Stage everything before single atomic rename.
 stage=args.out.with_name(args.out.name+".stage")
 if stage.exists():raise FileExistsError(stage)
 stage.mkdir(parents=True)
 for filename,data in (("public96_prompts.jsonl",public),("private96_tests.jsonl",private)):
  with (stage/filename).open("x") as f:
   for x in data:f.write(json.dumps(x,sort_keys=True,ensure_ascii=False)+"\n")
 manifest={"experiment":"EXP-021","source_sha256":SOURCE_SHA,"original_rows":378,"excluded_prior_tasks":24,"selected":96,"selection":"next 96 in historical EXP019B hash order excluding previous24","public_sha256":sha(stage/"public96_prompts.jsonl"),"private_sha256":sha(stage/"private96_tests.jsonl"),"source_test_inspected_for_selection":False}
 (stage/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
 stage.rename(args.out)
 print(json.dumps(manifest,indent=2))
if __name__=="__main__":main()
