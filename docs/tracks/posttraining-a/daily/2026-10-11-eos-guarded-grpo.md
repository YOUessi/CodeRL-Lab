# 2026-10-11：TRAIN-009C EOS-guarded learned-reward GRPO

**日期：2026-10-11。状态：代码、配置与单元测试已提交 GitHub，GPU正式验收必须等待TRAIN-009B的32提示终止率诊断与独占GPU释放。本实验只是受控smoke，不能冒称正式RLHF泛化结果，更不等于PPO。**

## 从真实失败出发，而非凭空构思

TRAIN-009A真实使用两份预训练权重（UltraChat正式QLoRA SFT及UltraFeedback全量NF4 Reward Model）进行on-policy GRPO：
- v2：96 Token生成4/4截断，mask导致梯度0、new-policy SHA与旧SFT相同；
- v3：256 Token生成8/8截断，smoke手动解除mask后确有2步非零梯度与新的Policy SHA，但都是未自然终止的截断答案，不能解释为高质量训练。

另一方面TRl1.14.1及Qwen3-1.7B-Base实测默认`eos_token_id=151643`、独立的`<|im_end|>=151645`，因此这里不能依赖字符串中含“结束语”就把完整性判为真。

## 固定新试验计划（未见本轮GPU结果前锁定）

- 初始SFT Actor仍来自 `TRAIN-007A`，sha `d8846aa5fb5fd6958d30a24611efd9fb99eb91469a60192937646b58557ce12d`；Reward Model来自全量`TRAIN-008A`，sha `586fb4cb21bb3408507d6e179d1bd858aca35d7c3e46b1dbef84234d3adbd28c`。
- 训练prompt仅来自UltraChat4096训练中预先确定的seed42前16条，排除所有Reward Model训练/验证及UltraChat SFT heldout重合（1条），并以4源SHA验证；不新增人造/挑过的“容易停”训练prompt，也不把原UltraChat SFT回答当RLHF标签。
- Online GRPO固定每组4条采样；`max_completion_length=512`、`max_prompt_length=240`，此预算与Reward Model的`max_length=768`相容且容纳完整生成（240+512+1=753）。
- **保留`mask_truncated_completions=true`**：绝不能把不完整答案作为正常policy gradient输入。
- Reward回调额外读取真实`completion_ids`：只有出现模型默认EOS Token151643时才使用冻结Reward Model打分并`tanh(raw/2)`；没有EOS的响应即使文本看似完整或含`<|im_end|>`也只得固定-1奖励（且由GRPO mask屏蔽其梯度）。没有全组至少一个真EOS则拒绝写成功摘要。
- 原TRL在`beta=0.02`时真实复制原始SFT LoRA为冻结`ref`；训练前逐参数确认392组ref/default一致且ref不可训练。
- **Smoke仅2 optimizer steps**，不允许在该实验直接跑全量4095prompt。必须实际看到非零`grad_norm`、新LoRA权重SHA异于原SFT、RM SHA完全未变、至少一条自然终止、真实rollout和KL统计，才算机械测试成功。
- 如果全2步的4/8条rollouts仍然没EOS：这属于有效的终止率负结果，保留日志与缺少训练更新，不强行降低“完成”要求。后续如需调EOS行为应使用独立EOS-aware SFT数据与无污染测试，不能靠反复调seed挑成功。
- 当前这条支线沿用TRAIN-009A的GRPORunner，GitHub 新配置 `configs/train009c_eos_guarded_grpo_smoke.yaml`、回调扩展 `learned_reward_grpo.py`、单卡入口 `scripts/run_train009c_eos_guarded_grpo_smoke.sh` 和 `tests/test_train009c_eos_guard.py`；全部位于GitHub独立分支，硬件仅用于真实CUDA。
- 结果如成功需上传仅聚合的数值和哈希，不上传prompt文本、raw completion或模型权重；如失败需保存异常与冻结身份，不伪造optimizer/accuracy。


## 22:05—22:10：严格自然EOS + 截断mask 的真机2步GRPO完整失败复盘

- Tang受独占锁在22:05:57+08实际运行代码 `827b95c6bceed804895907525c8c6851c4a9d1e6`；CPU 14项专项测试已通过，源16个训练prompt、真实TRAIN-007A SFT Adapter SHA、TRAIN-008A冻结RM Adapter SHA全部校验通过。测试配置固定2 optimizer steps、4rollouts/group，max_completion=512、mask_truncated=true、RM仅对真正EOS支付分数。
- 第1/2步实际生成共**8条回答，全部达到512token cap且自然EOS数为0**。TRL步数记录2/2但两步 `grad_norm=[0,0]`、KL=[0,0]、learned reward回调均输出固定截断惩罚-1（组内reward std=0），因此无法发生有效训练。
- 严格成功验收触发 `RuntimeError: ALL_TRUNCATED: no naturally terminated completion; refusing RLHF success`。**没有正式`run_summary.json`，没有声称任何新策略有效**。Trainer仍保存`checkpoint-1`/`checkpoint-2`的模型/optimizer作为失败现场，但其2步checkpoint-2 Adapter真实SHA与原SFT一致：`d8846aa5fb5fd6958d30a24611efd9fb99eb91469a60192937646b58557ce12d`，证明没有实际参数更新。
- 失败机器摘要已经保存在GitHub `results/train009c-eos-guarded-grpo-smoke/failed_result.json`，原始Tensor/log只在Tang；没有删除或覆盖旧失败。
- **为什么TRAIN-009B 512token终止率13/16与本轮0/8并不矛盾**：009B对每个提示按其索引单独固定`torch.manual_seed(42+i)`并采样一次，009C使用TRL独立group采样、全局状态与多代输入；并非同一条逐Token生成序列或相同RNG，实现内部可能还有采样/批处理区别。因此009B的样本比例是终止率诊断，不能当成“GRPO每一批自然EOS保证”。
- 工程判断：对于未自然EOS的所有样本，简单把截断mask打开会令训练梯度归零，关闭mask又只能证明截断响应有梯度，无法支撑完成答案质量。下一阶段需要从预先声明的**策略终止机制**入手（比如数据终止标签/SFT EOS补充、固定已存在的自然终止口径、增加多组独立rollout统计），而非继续在当前8条上挑seed或按私有奖励修改强度。
