"""EXP-023 prepare reproducible frozen-MBPP96 evaluators from EXP-021 source.

Run on Tang only, after arm training summary is finalized.
Creates arm-specific generator/scorer sources with adapter SHA frozen in code.
It does not open hidden test content or generate any model completions.
"""
from pathlib import Path
import argparse,json,hashlib
ARMS=("format","execution","full")
PREV_SHA="47e0d8d6687efe1965c2ba7ff88f064b56fe3d10907c742824b116abe1ffc8b4"
TRAIN_SOURCE_SHA="25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8"
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()
def prepare(workspace:Path,train:Path,arm:str):
 if arm not in ARMS:raise ValueError(arm)
 checkpoint=train/f"artifacts/exp023/grpo7b_{arm}_seed42"
 summary=json.loads((checkpoint/"summary.json").read_text())
 digest=sha(checkpoint/"adapter_model.safetensors")
 if (summary.get("status")!="real_GPU_GRPO_completed"
    or summary.get("reward_arm")!=arm
    or summary.get("global_step")!=32
    or summary.get("warm_start_source_sft_adapter_sha256")!=TRAIN_SOURCE_SHA
    or summary.get("saved_adapter_sha256")!=digest):
  raise ValueError("training arm failed frozen-source verification")
 src=workspace/"src/coderl_lab/analysis"
 a=src/"exp021_seed42_generate.py"
 b=src/"exp021_seed42_docker_score.py"
 gen=a.read_text()
 score=b.read_text()
 if gen.count(PREV_SHA)!=1 or score.count("from coderl_lab.analysis.exp021_seed42_generate import")!=1:
  raise ValueError("unexpected evaluation source; require explicit review")
 gen=gen.replace(PREV_SHA,digest)
 score=score.replace(f"from coderl_lab.analysis.exp021_seed42_generate import",f"from coderl_lab.analysis.exp023_{arm}_generate import")
 # scorer itself has GRPO_SHA imported from the arm-specific generator
 dest_gen=src/f"exp023_{arm}_generate.py"
 dest_score=src/f"exp023_{arm}_docker_score.py"
 for path,data in ((dest_gen,gen),(dest_score,score)):
  if path.exists():
   if path.read_text()!=data:raise FileExistsError(f"cannot replace existing evaluator: {path}")
  else:path.write_text(data)
 return {"arm":arm,"checkpoint":str(checkpoint),"adapter_sha256":digest,
         "generation_source_sha256":sha(dest_gen),"scorer_source_sha256":sha(dest_score)}
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--workspace",type=Path,required=True)
 p.add_argument("--train-root",type=Path,required=True)
 p.add_argument("--arm",choices=ARMS,required=True)
 a=p.parse_args()
 print(json.dumps(prepare(a.workspace,a.train_root,a.arm),indent=2))
if __name__=="__main__":main()
