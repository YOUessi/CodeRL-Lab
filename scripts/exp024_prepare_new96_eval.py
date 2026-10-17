"""EXP-024 create SHA-locked evaluator for new, nonoverlapping 96-task cohort.

Invoked on Tang after each independent 7B strict-reward training checkpoint is finalized.
Never reads hidden benchmark tests or evaluates model code.
"""
import argparse,hashlib,json
from pathlib import Path
EXPECTED_SOURCE="25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8"
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
 return h.hexdigest()
def prepare(workspace:Path,train:Path,seed:int):
 if seed not in (43,44):raise ValueError("only frozen independent seeds43/44")
 ckpt=train/f"artifacts/exp024/strict_seed{seed}"
 summary=json.loads((ckpt/"summary.json").read_text())
 adapter=sha(ckpt/"adapter_model.safetensors")
 if (summary.get("status")!="real_GPU_GRPO_completed" or summary.get("global_step")!=32
     or summary.get("warm_start_source_sft_adapter_sha256")!=EXPECTED_SOURCE
     or summary.get("saved_adapter_sha256")!=adapter
     or summary.get("reward_arm")!="strict"):
  raise ValueError("strict training adapter identity mismatch")
 src=workspace/"src/coderl_lab/analysis"
 gen=(src/"exp021_seed42_generate.py").read_text()
 score=(src/"exp021_seed42_docker_score.py").read_text()
 old="47e0d8d6687efe1965c2ba7ff88f064b56fe3d10907c742824b116abe1ffc8b4"
 if gen.count(old)!=1:raise ValueError("unreviewed source")
 gen=gen.replace(old,adapter).replace('m.get("experiment")!="EXP-021"','m.get("experiment")!="EXP-024"')
 gen=gen.replace('EXP-021-7B-SFT-VS-GRPO-MBPPPLUS','EXP-024-7B-SFT-VS-GRPO-MBPPPLUS')
 score=score.replace("from coderl_lab.analysis.exp021_seed42_generate import",
                     f"from coderl_lab.analysis.exp024_seed{seed}_generate import")
 for filename,contents in ((f"exp024_seed{seed}_generate.py",gen),
                           (f"exp024_seed{seed}_docker_score.py",score)):
  dest=src/filename
  if dest.exists():
   if dest.read_text()!=contents:raise FileExistsError(dest)
  else:dest.write_text(contents)
 return {"seed":seed,"model_adapter_sha256":adapter,"public_only_generator":f"exp024_seed{seed}_generate.py"}
if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--workspace",required=True,type=Path)
 p.add_argument("--train-root",required=True,type=Path)
 p.add_argument("--seed",required=True,type=int)
 a=p.parse_args()
 print(json.dumps(prepare(a.workspace,a.train_root,a.seed),indent=2))
