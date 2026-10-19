"""EXP030: one frozen-SFT trajectory per seed, audited 32/64/128 budgets.

No resume interface: Trainer checkpoints preserve optimizer/scheduler/RNG and
sampler metadata, but partial reward groups and TRL generation caches have not
been certified for exact replay. Restart in a fresh directory from frozen SFT.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import time
import traceback
from pathlib import Path
import yaml
from coderl_lab.train import exp024_7b_grpo as baseline
from coderl_lab.train import exp028_v3_7b_grpo as v3
from coderl_lab.train import exp027_matched_reward as matched
from coderl_lab.train import exp028_durable_reward as durable
from coderl_lab.train.train014b_reward_sandbox import validate_tests
from coderl_lab.analysis.exp028_protocol import sha as digest

MODEL=baseline.MODEL
REVISION=baseline.REVISION
FROZEN_SFT_SHA=baseline.FROZEN_SFT_SHA
read=baseline.read
SEEDS=(51,52,53)
CHECKPOINTS=(32,64,128)
FIXED_GRPO={'num_generations':4,'per_device_train_batch_size':4,
 'gradient_accumulation_steps':1,'max_completion_length':256,'learning_rate':1e-5,
 'temperature':.8,'top_p':.95,'beta':0.,'loss_type':'grpo','use_vllm':False,
 'num_iterations':1,'warmup_steps':0,'max_grad_norm':1.,'gradient_checkpointing':True,
 'adapter_sha256':FROZEN_SFT_SHA,'qlora_base_quantization':'nf4_double_quant_bf16'}


def validate_config(cfg,seed):
    if seed not in SEEDS or type(seed) is not int:raise ValueError('unregistered seed')
    if cfg.get('experiment')!={'id':'EXP030','coverage_arm':'narrow','reward_arm':'corrected'}:
        raise ValueError('experiment contract changed')
    if cfg.get('budget')!={'max_steps':128,'checkpoint_steps':[32,64,128]}:
        raise ValueError('budget must be one128-step trajectory with32/64/128 checkpoints')
    if cfg.get('grpo')!=FIXED_GRPO:raise ValueError('GRPO recipe changed')
    if cfg.get('lora')!={'r':8,'alpha':16,'target_modules':['q_proj','v_proj']}:
        raise ValueError('LoRA recipe changed')
    if cfg.get('reward',{}).get('wall_guard_seconds')!=60.:raise ValueError('wall guard changed')
    for key in ('data_manifest_sha256','gate_report_sha256','adapter_config_sha256'):
        h=cfg['assets'][key]
        if len(h)!=64 or any(c not in '0123456789abcdef' for c in h):raise ValueError('unfrozen asset: '+key)


def validate_paths(args):
    output=args.output_dir.absolute()
    if output.exists() or output.is_symlink():raise FileExistsError('refuse overwrite/resume: '+str(output))
    for asset in (args.data_dir.parent,args.sft_adapter,args.config,args.gate_report,args.cache_root):
        a=asset.resolve();o=output.resolve()
        if a==o or a in o.parents or o in a.parents:raise ValueError('output overlaps assets/cache')
    if args.cache_root.exists() or args.cache_root.is_symlink():raise FileExistsError('require new per-run cache directory')
    for asset in (args.data_dir.parent,args.sft_adapter):
        a=asset.resolve();c=args.cache_root.resolve()
        if c==a or a in c.parents or c in a.parents:raise ValueError('cache overlaps immutable assets')


def preflight(cfg,args):
    validate_config(cfg,args.seed);validate_paths(args)
    if args.data_dir.name!='narrow':raise ValueError('only frozen EXP028 narrow arm')
    data=args.data_dir.parent
    if digest(data/'manifest.json')!=cfg['assets']['data_manifest_sha256']:raise ValueError('data manifest hash')
    manifest=json.loads((data/'manifest.json').read_text())
    for rel,h in manifest['files'].items():
        p=(data/rel).resolve()
        if data.resolve() not in p.parents or digest(p)!=h:raise ValueError('frozen data drift/path escape')
    if digest(args.gate_report)!=cfg['assets']['gate_report_sha256']:raise ValueError('gate report hash')
    gate=json.loads(args.gate_report.read_text())
    if gate.get('passed') is not True or gate.get('status')!='gate_passed':raise ValueError('gate not passed')
    if gate.get('data_manifest_sha256')!=digest(data/'manifest.json'):raise ValueError('gate/data mismatch')
    if digest(args.sft_adapter/'adapter_model.safetensors')!=FROZEN_SFT_SHA:raise ValueError('frozen SFT hash')
    if digest(args.sft_adapter/'adapter_config.json')!=cfg['assets']['adapter_config_sha256']:raise ValueError('SFT config hash')
    adapter=json.loads((args.sft_adapter/'adapter_config.json').read_text())
    if (adapter.get('r')!=8 or adapter.get('lora_alpha')!=16 or
        set(adapter.get('target_modules',[]))!={'q_proj','v_proj'}):raise ValueError('adapter architecture')
    previous=baseline.FROZEN_DATA_EXPERIMENT
    baseline.FROZEN_DATA_EXPERIMENT='EXP028-fixed-coverage'
    try:result=baseline.preflight(cfg,args.data_dir,args.config)
    finally:baseline.FROZEN_DATA_EXPERIMENT=previous
    for row in read(args.data_dir/'train.jsonl'):validate_tests(row['test_cases'])
    return {**result,'experiment':'EXP030','training_seed':args.seed,
            'initial_sft_adapter_sha256':FROZEN_SFT_SHA,'checkpoint_steps':list(CHECKPOINTS),
            'execution_protocol':matched.protocol(),'assets':cfg['assets'],
            'source_sha256':digest(__file__),'same_trajectory_prefix':True,
            'resume_supported':False,'max_optimizer_steps':128}


def budget_reward(logfile,cache_root):
    callback=matched.reward_factory(logfile,'corrected',cache_root)
    groups=0
    def reward(prompts,completions,**kwargs):
        nonlocal groups
        if groups>=128:raise RuntimeError('hard reward group budget exceeded')
        result=callback(prompts,completions,**kwargs)
        groups+=1
        return result
    reward.__name__='exp030_full_v3_reward'
    return reward


def run(cfg,args,identity):
    config=cfg;g=cfg['grpo'];source_adapter=args.sft_adapter/'adapter_model.safetensors'
    started=time.monotonic()
    from transformers import TrainerCallback
    class AuditCallback(TrainerCallback):
        def on_step_begin(self,training,state,control,**kwargs):
            if state.global_step>=128:raise RuntimeError('hard optimizer budget exceeded')
        def on_step_end(self,training,state,control,**kwargs):
            step=int(state.global_step)
            if step>128:raise RuntimeError('optimizer budget overrun')
            durable.append(args.output_dir/'optimizer_steps.jsonl',{'global_step':step,'elapsed_seconds':time.monotonic()-started})
            control.should_save=step in CHECKPOINTS
            if step==128:control.should_training_stop=True
            return control
        def on_log(self,training,state,control,logs=None,**kwargs):
            durable.append(args.output_dir/'training_metrics.jsonl',{'global_step':state.global_step,'metrics':logs or {},'elapsed_seconds':time.monotonic()-started})
        def on_save(self,training,state,control,**kwargs):
            step=int(state.global_step);checkpoint=args.output_dir/f'checkpoint-{step}'
            if step not in CHECKPOINTS:raise RuntimeError('unregistered checkpoint')
            required=['adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json']
            missing=[name for name in required if not (checkpoint/name).is_file()]
            if missing:raise RuntimeError('incomplete training checkpoint: '+str(missing))
            sampler=trainer._get_train_sampler()
            sampler_info={'class':type(sampler).__module__+'.'+type(sampler).__name__,
                          'seed':args.seed,'global_step':step,'epoch':state.epoch,
                          'data_seed':args.seed,'dataloader_num_workers':0,
                          'resume_supported':False,'exact_replay_certified':False,
                          'boundary':'TRL buffered generations and sampler cursor replay not certified; fresh SFT rerun only'}
            if hasattr(sampler,'state_dict'):
                torch.save(sampler.state_dict(),checkpoint/'sampler_state.pt')
                sampler_info['state_dict_saved']=True
            else:sampler_info['state_dict_saved']=False
            durable.write_new(checkpoint/'sampler_metadata.json',sampler_info)
            attempts=read(args.output_dir/'reward_attempts.jsonl')
            if len(attempts)!=step*4:raise RuntimeError('reward/optimizer accounting mismatch')
            durable.write_new(checkpoint/'summary.json',{**identity,'global_step':step,
                'saved_adapter_sha256':digest(checkpoint/'adapter_model.safetensors'),
                'files_sha256':{name:digest(checkpoint/name) for name in required},
                'reward_candidates':len(attempts),'reward_attempt_log_sha256':digest(args.output_dir/'reward_attempts.jsonl'),
                'mean_reward':sum(x['reward'] for x in attempts)/len(attempts),
                'elapsed_seconds':time.monotonic()-started,
                'gpu_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                'gpu_peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20})
    import torch
    import peft,transformers,trl
    from datasets import Dataset
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    from trl import GRPOConfig,GRPOTrainer

    if not torch.cuda.is_available():raise RuntimeError("EXP030 requires real CUDA")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    tokenizer=AutoTokenizer.from_pretrained(
        MODEL,revision=REVISION,use_fast=True,local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    quant=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    backbone=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=REVISION,local_files_only=True,
        dtype=torch.bfloat16,device_map={"":0},
        quantization_config=quant,
    )
    backbone.config.use_cache=False
    model=PeftModel.from_pretrained(
        backbone,str(args.sft_adapter),adapter_name="default",is_trainable=True,
    )
    if not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("TRAIN017 must use frozen 7B QLoRA, not full BF16 backbone")
    initial_state=load_file(str(source_adapter),device="cpu")
    def assert_initial_sft_exact(peft_model):
        state=get_peft_model_state_dict(peft_model)
        if set(state)!=set(initial_state):
            raise ValueError("GRPO SFT initial LoRA adapter tensor keys differ")
        if any(not torch.equal(state[k].detach().to("cpu").to(initial_state[k].dtype),initial_state[k])
               for k in initial_state):
            raise ValueError("SFT LoRA model parameter values changed before optimizer")
    assert_initial_sft_exact(model)
    # This contains only the 48 TRAIN task/oracle rows. No heldout file content.
    train_rows=read(args.data_dir/"train.jsonl")
    # Apache Arrow cannot mix list-valued and string-valued x in a nested struct.
    # Keep the frozen tests byte-equivalent by encoding only this metadata
    # as JSON strings; the reward callback decodes before execution.
    for record in train_rows:
        record["test_cases"]=json.dumps(record["test_cases"],sort_keys=True)
    dataset=Dataset.from_list(train_rows)
    training=GRPOConfig(
        output_dir=str(args.output_dir),
        seed=int(args.seed),
        data_seed=int(args.seed),
        max_steps=128,
        per_device_train_batch_size=int(g["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(g["gradient_accumulation_steps"]),
        num_generations=int(g["num_generations"]),
        learning_rate=float(g["learning_rate"]),
        warmup_steps=int(g["warmup_steps"]),
        max_completion_length=int(g["max_completion_length"]),
        temperature=float(g["temperature"]),
        top_p=float(g["top_p"]),
        beta=float(g["beta"]),
        loss_type=g["loss_type"],
        num_iterations=int(g["num_iterations"]),
        scale_rewards="group",
        use_vllm=False,
        bf16=True,fp16=False,
        max_grad_norm=float(g["max_grad_norm"]),
        gradient_checkpointing=bool(g["gradient_checkpointing"]),
        logging_steps=1,logging_first_step=True,
        save_strategy="steps",save_steps=32,save_only_model=False,report_to="none",
        save_total_limit=None,
        disable_dropout=True,
        log_completions=False,
        dataloader_num_workers=0,
    )
    grader=budget_reward(args.output_dir/"reward_attempts.jsonl", args.cache_root)
    trainer=GRPOTrainer(
        model=model,
        reward_funcs=grader,
        args=training,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=None,
        callbacks=[AuditCallback()],
    )
    # TRL 1.14.1 converts quantized-model trainable LoRA to BF16
    # during trainer initialization. Restore FP32 BEFORE optimizer creation,
    # consistent with TRAIN-013 numeric precision control.
    assert_initial_sft_exact(trainer.model)
    trainable_params=[(n,p) for n,p in trainer.model.named_parameters() if p.requires_grad]
    if (not trainable_params
        or any(".lora_" not in n or ".default." not in n for n,p in trainable_params)):
        raise RuntimeError("unexpected trainable parameters outside 7B LoRA adapter")
    dtype_before=sorted({str(p.dtype) for _,p in trainable_params})
    for n,p in trainable_params:
        p.data=p.data.float()
    trainable=sorted({str(p.dtype) for _,p in trainable_params})
    if trainable!=["torch.float32"]:
        raise RuntimeError("EXP030 trainable LoRA not truly FP32 after TRL initialization")
    trained_count=sum(p.numel() for _,p in trainable_params)
    if trained_count!=2523136:
        raise RuntimeError(f"wrong SFT LoRA size (expected2523136 got{trained_count})")
    result=trainer.train()
    if trainer.state.global_step!=128:raise RuntimeError('incomplete optimization trajectory')
    for step in CHECKPOINTS:
        if not (args.output_dir/f'checkpoint-{step}'/'summary.json').is_file():raise RuntimeError('missing checkpoint summary')
    durable.write_new(args.output_dir/'summary.json',{**identity,'status':'real_GPU_GRPO_completed',
        'global_step':128,'train_runtime_seconds':time.monotonic()-started,
        'actual_train_metrics':dict(result.metrics),'versions':{'torch':torch.__version__,
        'trl':trl.__version__,'transformers':transformers.__version__,'peft':peft.__version__},
        'trainable_parameters':trained_count,'trainable_lora_dtype':trainable,
        'saved_adapter_sha256':digest(args.output_dir/'checkpoint-128'/'adapter_model.safetensors')})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','output-dir','sft-adapter','data-dir','gate-report','cache-root'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    args=p.parse_args();owned=False;safe_to_claim=False
    try:
        cfg=yaml.safe_load(args.config.read_text())
        validate_paths(args);safe_to_claim=True
        with v3.activated():
            identity=preflight(cfg,args)
            args.output_dir.mkdir(parents=True,exist_ok=False);owned=True
            args.cache_root.mkdir(parents=True,exist_ok=False)
            durable.write_new(args.output_dir/'identity.json',identity)
            run(cfg,args,identity)
    except BaseException as exc:
        if safe_to_claim and not owned and not args.output_dir.exists():
            args.output_dir.mkdir(parents=True,exist_ok=False);owned=True
        if owned:
            durable.write_new(args.output_dir/'failure.json',{'status':'aborted','error_type':type(exc).__name__,
                'reason':str(exc),'traceback':traceback.format_exc(),'resume_supported':False})
        raise

if __name__=='__main__':main()
