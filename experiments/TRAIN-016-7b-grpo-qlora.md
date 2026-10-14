# TRAIN-016：回归 Qwen2.5-7B-Instruct，真实 7B GRPO 代码正确性强化学习

**冻结日期：2026-10-15（唐 Tang RTX4090 16GB），7B GPU冒烟前登记。**

## 项目定位与纠偏
主研究模型始终应为 `Qwen/Qwen2.5-7B-Instruct`，已有 TRAIN-011 正式4096样本 SFT 和 TRAIN-012/013 7B BF16/FP32 DPO 产物。之前 TRAIN-014/014B 使用1.5B仅能视为 **GRPO框架、Docker奖励、loss/梯度与模型保存的工程预实验**，不能把1.5B训练效果当作CodeRL-Lab主科研结论，更不能与7B同模型SFT/DPO直接进行因果对照。现在停止以1.5B为主模型的新训练。

## 相同7B模型与初始化
- Backbone: `Qwen/Qwen2.5-7B-Instruct` revision `a09a35458c702b33eeacc393d103063234e8bc28`。
- 实验学习起点不是随机LoRA、更不是1.5B权重，而是 **TRAIN-011 已冻结并完整训练的 7B SFT Adapter**：`25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8`。背骨使用 NF4 4-bit+double quant，BF16前向，原LoRA结构 r8/alpha16/q_proj/v_proj，GRPOTrainer必须接受已训练的 `PeftModel`（`peft_config=None`），否则会重新创建随机Adapter而破坏SFT继承。
- 为缓解已发现的TRAIN-013低精度梯度问题，检查GRPOTrainer初始化后可训练LoRA的真实dtype，并**仅将可训练LoRA参数转回 FP32**（底座不变）；检查初始LoRA tensor与TRAIN-011源Adapter相等、总数和名称正确，再创建优化器。保存时报告Adapter SHA与初始数值变化，若没有任何参数变化不能宣称学习成功。
- 同一个冻结 `TRAIN-014B` 的36训练/12留出算法代码任务可供**7B算法管线和受控新模型切换工程验收**，sha:
  - train `c076b6a3d685ed1b3b4df20e944be78064f069371a1ef177ed8b2e6342eabfe7`
  - holdout public `009dc276e718d4ec600faa3b885f0f934c225d0dd0637d6d8d3b468caaa6d834`
  - holdout oracle `cfcd2467b8c43fa084a63f7fcf380fe58bd6d483b2530dbbab358b10cd6a822e`。
- **科研诚实边界：** TRAIN-014B的12题holdout分数已被看过，是7B工程迁移/探索性比较，不再算严格全新盲测；后续要另备真实新题并提前冻结。

## GPU资源与逐级验证
- Tang单GPU显存16GB（RTX4090 Laptop），不满足无量化7B全量GRPO。先用 **7B QLoRA**、`GRPOTrainer` `use_vllm=False`、`beta=0`（无需独立reference backbone）、单GPU、`num_generations=2`、`batch_size=2`、`max_completion_length=160`、组内temperature0.8/top_p0.95、LR1e-5、seed42、gradient checkpointing，所有开关实测后写入日志。
- Smoke：真实2个优化步骤→完成模型生成→经沙箱执行真实Docker 10~40条测试→部分正确比例奖励→逐组std/实际非零梯度→训练后Adapter SHA/显存。失败或OOM原样保留，不更改原目录，另设v2冻结配置。
- Smoke通过后再决定单GPU正式16个优化步是否可行；如果组内奖励全相同（正如TRAIN-014 v1），明确记录零梯度并先做难度/奖励区分度诊断，**绝不把零梯度当强化学习提升**。
- GPU竞争用独占文件锁，源数据/答案与代码只存Tang，GitHub作为配置/脚本/测试/日期日志与可复现统计的唯一事实源；Docker镜像 `python:3.11-slim@sha256:d3846bce6e97d0ef5c32fc023300d5e39251ad791e6fdfaee41d2ba75246766e`，非root禁网readonly，未知基础设施错误必须中断。
- 7B真实效果应以**同一7B的冻结SFT vs 7B GRPO**生成+执行为主要对照，单次与多次采样同参数、同seed。7B DPO为另一独立模型状态，需同数据同采样预算三组评测才能比较；不能拿之前1.5B结果与7B混合称为训练效应。

## 分支
`feat/train016-7b-grpo-qlora` 为A线新分支，仅沿用已经工程验证的GRPO/Docker奖励模块，不合并或修改独立全栈控制台和已有负结果。